import os
import json
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import subprocess

app = FastAPI(title="SCRUMBENCH Experiment Dashboard")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

@app.get("/")
async def read_index():
    return FileResponse("index.html")

class ExperimentSummary(BaseModel):
    id: str
    name: str
    timestamp: str
    total_tasks: int
    successful_tasks: int
    mean_ics: float
    status: str

# Track running experiments
running_experiments: Dict[str, str] = {}

@app.get("/baselines")
def list_baselines():
    from core.runner import get_baseline_configs
    configs = get_baseline_configs()
    return [{"id": k, "name": v.name, "description": v.description} for k, v in configs.items()]

@app.get("/experiments", response_model=List[ExperimentSummary])
def list_experiments():
    experiments = []
    seen_ids = set()
    
    from core.runner import get_baseline_configs
    baselines = get_baseline_configs()
    
    # 1. Add running experiments first
    for run_id, baseline_id in running_experiments.items():
        config = baselines.get(baseline_id)
        total_tasks = sum(t.num_instances * (len(t.difficulty_levels) if t.difficulty_levels else 1) for t in config.tasks) if config else 0
        
        # Live stats from manifest and workspaces
        manifest_file = OUTPUT_DIR / run_id / "task_manifest.json"
        manifest = []
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text())
                total_tasks = len(manifest)
            except: pass
        
        successful = 0
        ics_sum = 0.0
        finished_count = 0
        workspaces_dir = OUTPUT_DIR / run_id / "workspaces"
        if workspaces_dir.exists():
            for task_dir in workspaces_dir.iterdir():
                if (task_dir / "trace.json").exists():
                    finished_count += 1
                    diag_file = task_dir / "diagnostics.json"
                    if diag_file.exists():
                        try:
                            diag = json.loads(diag_file.read_text())
                            if diag.get("outcome") == "success": successful += 1
                            ics_sum += diag.get("integration_competence_score", 0.0)
                        except: pass
        
        experiments.append(ExperimentSummary(
            id=run_id,
            name=f"{run_id} (Running...)",
            timestamp=datetime.now().isoformat(),
            total_tasks=total_tasks,
            successful_tasks=successful,
            mean_ics=ics_sum / finished_count if finished_count > 0 else 0.0,
            status="running"
        ))
        seen_ids.add(run_id)

    if not OUTPUT_DIR.exists():
        return experiments
    
    for run_dir in OUTPUT_DIR.iterdir():
        if not run_dir.is_dir() or run_dir.name in seen_ids:
            continue
        
        results_file = run_dir / "results.json"
        
        # Check if it's currently running (even if not started by dashboard)
        # Simple heuristic: directory exists but no final results.json or it's recently modified
        is_running = not results_file.exists() or (datetime.now().timestamp() - run_dir.stat().st_mtime < 60)
        
        status = "completed"
        if results_file.exists():
            try:
                data = json.loads(results_file.read_text())
                successful = data.get("successful_tasks", 0)
                total = data.get("total_tasks", 0)
                ics = data.get("mean_ics", 0.0)
                timestamp = data.get("start_time", "")
            except:
                continue
        else:
            # Running via CLI or just started
            status = "running"
            successful = 0
            ics = 0.0
            total = 0
            timestamp = datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat()
            
        # For non-dashboard running experiments, scan workspaces
        if status == "running" or is_running:
            workspaces_dir = run_dir / "workspaces"
            if workspaces_dir.exists():
                finished_count = 0
                ics_sum = 0.0
                successful_count = 0
                for task_dir in workspaces_dir.iterdir():
                    if (task_dir / "trace.json").exists():
                        finished_count += 1
                        diag_file = task_dir / "diagnostics.json"
                        if diag_file.exists():
                            try:
                                diag = json.loads(diag_file.read_text())
                                if diag.get("success"): successful_count += 1
                                ics_sum += diag.get("integration_competence_score", 0.0)
                            except: pass
                
                # Update if we found more than results.json showed
                successful = max(successful, successful_count)
                if finished_count > 0:
                    ics = ics_sum / finished_count
                if status == "running":
                    status = "running" # Keep it running

        experiments.append(ExperimentSummary(
            id=run_dir.name,
            name=run_dir.name,
            timestamp=timestamp,
            total_tasks=total or 10, # Fallback
            successful_tasks=successful,
            mean_ics=ics,
            status=status
        ))
                
    # Sort by timestamp descending
    experiments.sort(key=lambda x: x.timestamp, reverse=True)
    return experiments

@app.get("/experiments/{run_id}")
def get_experiment(run_id: str):
    results_file = OUTPUT_DIR / run_id / "results.json"
    workspaces_dir = OUTPUT_DIR / run_id / "workspaces"
    
    # Base data
    data = None
    if results_file.exists():
        try:
            data = json.loads(results_file.read_text())
        except: pass
            
    if data is None:
        # Try to build skeleton from config.json or folder existence
        experiment_dir = OUTPUT_DIR / run_id
        if not experiment_dir.exists():
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        config_file = experiment_dir / "config.json"
        total_tasks = 10 # Default fallback
        description = "Experiment in progress..."
        if config_file.exists():
            try:
                conf = json.loads(config_file.read_text())
                total_tasks = sum(t.get("num_instances", 1) * (len(t.get("difficulty_levels", ["easy"])) if t.get("difficulty_levels") else 1) for t in conf.get("tasks", []))
                description = conf.get("description", description)
            except: pass
        elif run_id in running_experiments:
            from core.runner import get_baseline_configs
            baselines = get_baseline_configs()
            baseline_id = running_experiments[run_id]
            config = baselines.get(baseline_id)
            if config:
                total_tasks = sum(t.num_instances * (len(t.difficulty_levels) if t.difficulty_levels else 1) for t in config.tasks)
                description = config.description

        data = {
            "experiment_name": run_id,
            "description": description,
            "total_tasks": total_tasks,
            "successful_tasks": 0,
            "mean_ics": 0.0,
            "task_results": [],
            "results": []
        }


    # Read task manifest to get full planned task list
    manifest_file = OUTPUT_DIR / run_id / "task_manifest.json"
    manifest = []
    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text())
        except: pass
    
    # Build comprehensive task list from manifest
    if manifest and workspaces_dir.exists():
        # Clear existing task_results to rebuild from manifest
        data["task_results"] = []
        
        for task_spec in manifest:
            instance_id = task_spec["instance_id"]
            task_dir = workspaces_dir / instance_id
            trace_file = task_dir / "trace.json"
            diag_file = task_dir / "diagnostics.json"
            
            # Determine task status
            if trace_file.exists():
                # Completed task
                diag = json.loads(diag_file.read_text()) if diag_file.exists() else {}
                data["task_results"].append({
                    "instance_id": instance_id,
                    "template": task_spec.get("template", "unknown"),
                    "difficulty": task_spec.get("difficulty", "unknown"),
                    "success": diag.get("outcome") == "success",
                    "outcome": diag.get("outcome", "unknown"),
                    "integration_competence_score": diag.get("integration_competence_score", 0.0),
                    "status": "completed"
                })
            elif task_dir.exists():
                # In progress (workspace exists but no trace yet)
                data["task_results"].append({
                    "instance_id": instance_id,
                    "template": task_spec.get("template", "unknown"),
                    "difficulty": task_spec.get("difficulty", "unknown"),
                    "success": False,
                    "outcome": "in_progress",
                    "integration_competence_score": 0.0,
                    "status": "in_progress"
                })
            else:
                # Todo (not started yet)
                data["task_results"].append({
                    "instance_id": instance_id,
                    "template": task_spec.get("template", "unknown"),
                    "difficulty": task_spec.get("difficulty", "unknown"),
                    "success": False,
                    "outcome": "pending",
                    "integration_competence_score": 0.0,
                    "status": "todo"
                })
        
        # Update summary stats
        completed_tasks = [r for r in data["task_results"] if r["status"] == "completed"]
        if completed_tasks:
            data["successful_tasks"] = sum(1 for r in completed_tasks if r["success"])
            data["mean_ics"] = sum(r["integration_competence_score"] for r in completed_tasks) / len(completed_tasks)
    elif workspaces_dir.exists():
        # Fallback: scan workspaces if no manifest (for old experiments)
        seen_instances = {r["instance_id"] for r in data.get("task_results", [])}
        for task_dir in workspaces_dir.iterdir():
            if not task_dir.is_dir() or task_dir.name in seen_instances:
                continue
            
            trace_file = task_dir / "trace.json"
            diag_file = task_dir / "diagnostics.json"
            if trace_file.exists():
                diag = json.loads(diag_file.read_text()) if diag_file.exists() else {}
                meta_file = task_dir / "metadata.json"
                difficulty = "unknown"
                if meta_file.exists():
                    try:
                        meta = json.loads(meta_file.read_text())
                        difficulty = meta.get("difficulty", "unknown")
                    except: pass

                data["task_results"].append({
                    "instance_id": task_dir.name,
                    "success": diag.get("success", False),
                    "integration_competence_score": diag.get("integration_competence_score", 0.0),
                    "difficulty": difficulty,
                    "status": "completed"
                })
                
                count = len(data["task_results"])
                if count > 0:
                    data["successful_tasks"] = sum(1 for r in data["task_results"] if r.get("success"))
                    data["mean_ics"] = sum(r["integration_competence_score"] for r in data["task_results"]) / count

    return data

@app.get("/experiments/{run_id}/tasks/{instance_id}")
def get_task_details(run_id: str, instance_id: str):
    task_dir = OUTPUT_DIR / run_id / "workspaces" / instance_id
    trace_file = task_dir / "trace.json"
    diag_file = task_dir / "diagnostics.json"
    kanban_file = task_dir / "starter" / "kanban.json"
    metadata_file = task_dir / "metadata.json"
    
    trace = json.loads(trace_file.read_text()) if trace_file.exists() else {"turns": [], "messages_exchanged": 0}
    diagnostics = json.loads(diag_file.read_text()) if diag_file.exists() else {}
    initial_kanban = json.loads(kanban_file.read_text()) if kanban_file.exists() else []
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
    
    # Enhance trace with detailed turn information
    if trace_file.exists() and "turns" in trace:
        # Load agent conversation logs to get tool calls and token usage
        enhanced_turns = []
        for turn in trace["turns"]:
            enhanced_turn = turn.copy()
            
            # Try to load agent-specific log file for detailed info
            agent_id = turn.get("agent", turn.get("agent_id", "unknown"))
            agent_log_file = task_dir / f"agent_{agent_id}_log.json"
            
            # For now, we'll work with what we have in the trace
            # The trace already has basic info, we can enhance it with more details
            enhanced_turns.append(enhanced_turn)
        
        trace["turns"] = enhanced_turns
    
    # Calculate token usage summary from trace
    token_summary = {}
    if "total_tokens" in trace:
        token_summary = trace["total_tokens"]
    
    # Check if this is a ScrumBan experiment (has real kanban.json)
    is_scrumban = kanban_file.exists()
    
    # For non-ScrumBan: create experiment-level Kanban showing all tasks
    if not initial_kanban:
        # Read manifest to get all tasks
        manifest_file = OUTPUT_DIR / run_id / "task_manifest.json"
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text())
                workspaces_dir = OUTPUT_DIR / run_id / "workspaces"
                
                # Build experiment-level Kanban
                initial_kanban = []
                for i, task_spec in enumerate(manifest):
                    task_id = task_spec["instance_id"]
                    task_workspace = workspaces_dir / task_id
                    task_trace = task_workspace / "trace.json"
                    
                    # Determine status
                    if task_trace.exists():
                        status = "done"
                    elif task_workspace.exists():
                        status = "in_progress"
                    else:
                        status = "todo"
                    
                    initial_kanban.append({
                        "id": task_id,
                        "title": task_id.replace("_", " ").title(),
                        "description": f"{task_spec.get('template', 'task')} - {task_spec.get('difficulty', 'unknown')}",
                        "status": status,
                        "difficulty": task_spec.get("difficulty", "unknown")
                    })
            except: pass

    res = {
        "trace": trace,
        "diagnostics": diagnostics,
        "initial_kanban": initial_kanban,
        "metadata": metadata,
        "token_summary": token_summary
    }
    if not trace_file.exists():
        res["status"] = "pending"
    return res

@app.get("/experiments/{run_id}/analytics")
def get_experiment_analytics(run_id: str):
    """Get analytics data for charts and visualizations."""
    workspaces_dir = OUTPUT_DIR / run_id / "workspaces"
    
    if not workspaces_dir.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Collect data from all task instances
    tasks_by_difficulty = {"easy": [], "medium": [], "hard": []}
    token_timeline = []
    agent_activity = {}
    
    for task_dir in sorted(workspaces_dir.iterdir()):
        if not task_dir.is_dir():
            continue
            
        trace_file = task_dir / "trace.json"
        diag_file = task_dir / "diagnostics.json"
        metadata_file = task_dir / "metadata.json"
        
        if not trace_file.exists():
            continue
            
        try:
            trace = json.loads(trace_file.read_text())
            diag = json.loads(diag_file.read_text()) if diag_file.exists() else {}
            metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
            
            difficulty = metadata.get("difficulty", "unknown")
            success = diag.get("outcome") == "success"
            
            # Group by difficulty
            if difficulty in tasks_by_difficulty:
                tasks_by_difficulty[difficulty].append({
                    "instance_id": task_dir.name,
                    "success": success,
                    "ics": diag.get("integration_competence_score", 0.0)
                })
            
            # Token timeline
            if "total_tokens" in trace:
                total = sum(
                    tokens.get("prompt", 0) + tokens.get("completion", 0)
                    for tokens in trace["total_tokens"].values()
                )
                token_timeline.append({
                    "instance_id": task_dir.name,
                    "tokens": total
                })
            
            # Agent activity
            for turn in trace.get("turns", []):
                agent = turn.get("agent", "unknown")
                if agent not in agent_activity:
                    agent_activity[agent] = {"turns": 0, "tokens": 0}
                agent_activity[agent]["turns"] += 1
                
        except Exception as e:
            continue
    
    # Calculate success rates by difficulty
    success_by_difficulty = {}
    for diff, tasks in tasks_by_difficulty.items():
        if tasks:
            success_rate = sum(1 for t in tasks if t["success"]) / len(tasks) * 100
            success_by_difficulty[diff] = {
                "total": len(tasks),
                "successful": sum(1 for t in tasks if t["success"]),
                "success_rate": success_rate,
                "mean_ics": sum(t["ics"] for t in tasks) / len(tasks) if tasks else 0
            }
    
    return {
        "success_by_difficulty": success_by_difficulty,
        "token_timeline": token_timeline,
        "agent_activity": agent_activity
    }

@app.post("/run")
async def run_experiment(baseline: str, background_tasks: BackgroundTasks):
    from core.runner import get_baseline_configs, ExperimentRunner
    
    baselines = get_baseline_configs()
    if baseline not in baselines:
        raise HTTPException(status_code=400, detail=f"Unknown baseline: {baseline}")
    
    config = baselines[baseline]
    config.output_dir = str(OUTPUT_DIR) # Ensure it writes to dashboard directory
    run_id = config.name
    
    if run_id in running_experiments:
        return {"status": "already_running", "run_id": run_id}

    def run_job():
        try:
            running_experiments[run_id] = baseline
            runner = ExperimentRunner(config)
            runner.run()
        finally:
            if run_id in running_experiments:
                del running_experiments[run_id]
        
    background_tasks.add_task(run_job)
    return {"status": "started", "run_id": run_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
