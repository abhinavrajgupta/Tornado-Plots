import subprocess

module_name = input("Enter the module name (e.g., 'M35'): ")
source_path = f"Modules/{module_name}/"

command = [
    "python", "detect.py",
    "--weights", "best.pt",
    "--img", "640",
    "--conf", "0.6",
    "--source", source_path,
    "--project", source_path,
    "--name", "ResultsArrowPlots",
    "--save-txt"
]

try:
    subprocess.run(command, check=True)
    print("Detection completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"Error during detection: {e}")
