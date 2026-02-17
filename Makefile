# Makefile for GDSKG MCP Tray Application

PYTHON = python3
PIP = .venv/bin/pip
PYINSTALLER = .venv/bin/pyinstaller

# Detect OS
ifeq ($(OS),Windows_NT)
    EXE_NAME = gdskg-mcp.exe
    SEP = ;
else
    EXE_NAME = gdskg-mcp
    SEP = :
    UNAME_S := $(shell uname -s)
endif

.PHONY: all clean venv install build

all: build

venv:
	test -d .venv || $(PYTHON) -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build: install
	@echo "Building GDSKG MCP Tray Application..."
	$(PYINSTALLER) --noconsole --onefile \
		--name "$(EXE_NAME)" \
		--add-data "mcp_server/icon.png$(SEP)mcp_server" \
		--add-data "mcp_server/server.py$(SEP)mcp_server" \
		--add-data "core$(SEP)core" \
		--add-data "analysis$(SEP)analysis" \
		--add-data "plugins$(SEP)plugins" \
		--hidden-import "mcp.server.fastmcp" \
		--hidden-import "pystray" \
		--hidden-import "PIL" \
		mcp_server/tray.py
	@echo "Build complete! Portable executable found in dist/$(EXE_NAME)"

clean:
	rm -rf build dist *.spec
	@echo "Cleaned build artifacts."
