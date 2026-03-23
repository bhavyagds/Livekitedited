import multiprocessing
import sys
import logging

def test_spawn():
    print("In child process, importing src.agents.elena...")
    try:
        import src.agents.elena
        print("Child: Imported successfully!")
    except Exception as e:
        print(f"Child: Import failed: {e}")

if __name__ == '__main__':
    print("Setting spawn method")
    multiprocessing.set_start_method('spawn')
    print("Starting child process")
    p = multiprocessing.Process(target=test_spawn)
    p.start()
    print("Joining child process")
    p.join()
    print(f"Done, exit code {p.exitcode}")
