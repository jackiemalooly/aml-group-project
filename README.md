# University of Surrey Assigmment Template Repository 🦌
Hello! Tired of setting up repositories for your projects? Here's a template repository for all the projects in the University of Surrey CVRML MSc. Feel free to use this repository as a starting template for assignments, thesis, or any other projects you have in mind 🫶

## Directory Layout
```bash
project_name
├── LICENSE
├── README.md
├── Surrey_CSEE_Thesis_Template
└── code
```
## LaTeX Setup
This comes with a University of Surrey Faculty of Engineering and Physical Sciences thesis template created by Aaron and Alireza on [Overleaf](https://www.overleaf.com/latex/templates/surrey-feps-confirmation-report-template/kffgbyxwcrbg). I recommend setting up a local distribution of Tex Live on your machine because Overleaf can time out when compiling large documents.

Here are explicit instructions on how to set up LaTeX on your machine:

### 1. Install TeX Live
You can download TeX Live [here](https://www.tug.org/texlive/).
For macOS users, you can install Tex Live using Homebrew. This takes a while to install, so be patient 🐢.
```bash
brew install --cask mactex
```

### 2. Download Visual Studio Code
```bash
brew install --cask visual-studio-code
```

### 3. Install the LaTeX Workshop Extension in Visual Studio Code
```bash
code --install-extension james-yu.latex-workshop
```
### 4. Edit User Settings in Visual Studio Code
Press `Cmd + shift + p,` and type `preferences: Open User Settings (JSON)` to open the `settings.json` file. Add the following `latex-workshop` settings:
```json
"latex-workshop.latex.tools": [
    {
        "name": "latexmk",
        "command": "latexmk",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "-pdf",
            "-outdir=%OUTDIR%",
            "%DOC%"
        ],
        "env": {}
    },
    {
        "name": "xelatex",
        "command": "xelatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ],
        "env": {}
    },
    {
        "name": "pdflatex",
        "command": "pdflatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ],
        "env": {}
    },
    {
        "name": "bibtex",
        "command": "bibtex",
        "args": [
            "%DOCFILE%"
        ],
        "env": {}
    }
],
"latex-workshop.latex.recipes": [
    {
        "name": "pdfLaTeX",
        "tools": [
            "pdflatex"
        ]
    },
    {
        "name": "latexmk 🔃",
        "tools": [
            "latexmk"
        ]
    },
    {
        "name": "xelatex",
        "tools": [
            "xelatex"
        ]
    },
    {
        "name": "pdflatex ➞ bibtex ➞ pdflatex`×2",
        "tools": [
            "pdflatex",
            "bibtex",
            "pdflatex",
            "pdflatex"
        ]
    },
    {
        "name": "xelatex ➞ bibtex ➞ xelatex`×2",
        "tools": [
            "xelatex",
            "bibtex",
            "xelatex",
            "xelatex"
        ]
    }
],
```

## Code Setup
Given that most of the projects in the CVRML courses are in Python, I have included a `code` directory with a `poetry` setup with the necessary dependencies.

To install the dependencies, run the following commands in the code directory:
```bash
poetry shell
poetry install
```
