import multiprocessing
import time
import sys
import os

# A function that does some work and updates the progress
def worker(task_number, progress_dict):
    for i in range(5):  # Simulate some work with sleep
        time.sleep(1)
        progress_dict[task_number] = (i + 1) / 5 * 100  # Update progress percentage

if __name__ == "__main__":
    total_tasks = 4
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()  # Shared dictionary to track progress

    processes = []
    for i in range(total_tasks):
        progress_dict[i] = 0  # Initialize progress for each process
        p = multiprocessing.Process(target=worker, args=(i, progress_dict))
        processes.append(p)
        p.start()

    # Continuously print the progress
    try:
        while any(p.is_alive() for p in processes):
            # Clear the terminal (optional)
            os.system('cls' if os.name == 'nt' else 'clear')

            # Print progress for each task on its own line
            for task_number in range(total_tasks):
                progress = progress_dict[task_number]
                print(f"Task {task_number + 1}/{total_tasks} - Progress: {progress:.2f}%")

            time.sleep(0.5)  # Refresh rate

    except KeyboardInterrupt:
        print("\nProcess interrupted.")
    
    for p in processes:
        p.join()

    print("\nAll tasks are complete!")
