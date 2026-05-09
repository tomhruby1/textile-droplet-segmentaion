import subprocess

with open("version") as f:
    version = f.read()

args = [
    "nuitka",
    # "--standalone",
    "--deployment",
    "--no-pyi-file",
    "--enable-plugin=pyside6",
    "--mode=app",
    f"--product-version={version}",
    "--output-filename=DropSeg",
    "src/segmentation_editor.py"
]

subprocess.run(args, check=True)