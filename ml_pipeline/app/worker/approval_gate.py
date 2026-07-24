import os
import time
import json
import logging

logger = logging.getLogger(__name__)

def check_user_permission(task_id: str, results: list) -> bool:
    """
    Temporary approval gate. Writes the result payload to a local JSON file.
    Blocks the background thread until the user creates 'approve_<task_id>.txt'
    containing 'YES' or 'NO'.
    """
    # Use paths relative to the worker folder or workspace
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    approval_file_path = os.path.join(base_dir, f"pending_approval_{task_id}.json")
    response_file_path = os.path.join(base_dir, f"approve_{task_id}.txt")
    
    # Write payload for user inspection
    try:
        with open(approval_file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info("[APPROVAL GATE] Results payload written to %s", approval_file_path)
    except Exception as err:
        logger.error("[APPROVAL GATE] Failed to write payload file: %s", err)
        return True # Fallback to sending if we can't write
        
    print(f"\n============================================================\n")
    print(f"[APPROVAL REQUIRED] Task {task_id} results are ready.")
    print(f"Payload written to: {approval_file_path}")
    print(f"To approve, create a file named {response_file_path} containing 'YES'.")
    print(f"To reject, create a file containing 'NO'.")
    print(f"\n============================================================\n")
    
    # Loop and wait for response file
    start_time = time.time()
    timeout = 3600  # 1 hour
    while time.time() - start_time < timeout:
        if os.path.exists(response_file_path):
            try:
                time.sleep(0.5) # ensure file write finished
                with open(response_file_path, "r", encoding="utf-8") as rf:
                    choice = rf.read().strip().upper()
                
                # Cleanup
                if os.path.exists(response_file_path):
                    os.remove(response_file_path)
                if os.path.exists(approval_file_path):
                    os.remove(approval_file_path)
                    
                if choice == "YES":
                    logger.info("[APPROVAL GATE] User approved payload for Task %s", task_id)
                    return True
                elif choice == "NO":
                    logger.info("[APPROVAL GATE] User rejected payload for Task %s", task_id)
                    return False
            except Exception as e:
                logger.error("[APPROVAL GATE] Error processing response: %s", e)
        time.sleep(2)
        
    # Timeout cleanup
    if os.path.exists(approval_file_path):
        try:
            os.remove(approval_file_path)
        except Exception:
            pass
    logger.warning("[APPROVAL GATE] Timeout waiting for approval on Task %s. Payload NOT sent.", task_id)
    return False
