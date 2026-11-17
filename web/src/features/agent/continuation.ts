type ContinuationState = {
  state: string;
  runtime_availability?: string;
  dispatch_runtime_availability?: string;
  cancellation_requested?: boolean;
};

export function canContinueTask(task: ContinuationState): boolean {
  return ["succeeded", "partial", "waiting_input"].includes(task.state)
    && !task.cancellation_requested
    && (!task.runtime_availability || task.runtime_availability === "current")
    && (!task.dispatch_runtime_availability || task.dispatch_runtime_availability === "current");
}
