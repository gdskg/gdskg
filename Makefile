# Makefile for GDSKG MCP Tray Application

PYTHON = python3
PIP = .venv/bin/pip
PYINSTALLER = .venv/bin/pyinstaller
VERSION = $(shell cat VERSION)

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

docker-build:
	docker build -t gdskg:$(VERSION) .
	docker tag gdskg:$(VERSION) gdskg:latest
	@echo "Docker image built: gdskg:$(VERSION) and gdskg:latest"

docker-run:
	@echo "Checking for existing GDSKG containers..."
	@docker stop gdskg-server >/dev/null 2>&1 || true
	@docker rm gdskg-server >/dev/null 2>&1 || true
	@echo "Starting GDSKG MCP Server in Docker (port 8015)..."
	docker run -d \
		--name gdskg-server \
		-p 8015:8015 \
		-v $(shell pwd)/gdskg_graph:/app/gdskg_graph \
		-e GITHUB_PAT \
		gdskg:latest
	@echo "Server is running in background. Logs available via 'docker logs -f gdskg-server'"
