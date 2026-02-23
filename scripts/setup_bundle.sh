#!/bin/bash

# GDSKG Local Setup Script - Interactive Bundle Builder
# This script builds a Docker image with a pre-indexed knowledge graph,
# mirroring the logic used in the GitHub Actions template.

set -e

# Colors for better UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}   GDSKG Interactive Knowledge Graph Builder       ${NC}"
echo -e "${BLUE}===================================================${NC}"
echo ""

# 1. Ask for repositories
echo -e "${YELLOW}Step 1: Repository Configuration${NC}"
echo "Enter the GitHub repositories you want to index (space-separated)."
echo "Format: owner/repo or full URL"
read -p "Repositories: " REPOS

if [ -z "$REPOS" ]; then
    echo -e "${RED}Error: At least one repository is required.${NC}"
    exit 1
fi

# 2. Ask for Auth Token
echo ""
echo -e "${YELLOW}Step 2: Authentication${NC}"
echo "If any repositories are private, or to avoid GitHub rate limits,"
echo "please provide a Personal Access Token (PAT)."
read -sp "GitHub Token (optional): " GITHUB_TOKEN
echo ""

# 3. Ask for destination image name
echo ""
echo -e "${YELLOW}Step 3: Image Configuration${NC}"
read -p "Enter name for the final Docker image [gdskg-bundled]: " IMAGE_NAME
IMAGE_NAME=${IMAGE_NAME:-gdskg-bundled}

# 4. Build base image
echo ""
echo -e "${YELLOW}Step 4: Preparing Base Image${NC}"
echo "Building 'gdskg:latest' from current source to ensure parity..."
docker build -t gdskg:latest .

# 5. Build Graph
echo ""
echo -e "${YELLOW}Step 5: Building Knowledge Graph${NC}"
TEMP_GRAPH_DIR=$(mktemp -d)
TEMP_VECTOR_DIR=$(mktemp -d)

# Ensure they are writable by docker
chmod 777 "$TEMP_GRAPH_DIR" "$TEMP_VECTOR_DIR"

for REPO in $REPOS; do
    echo -e "${BLUE}Indexing $REPO...${NC}"
    
    # Formulate repo URL
    if [[ "$REPO" != http* ]]; then
        if [ -n "$GITHUB_TOKEN" ]; then
            TARGET_REPO="https://x-access-token:${GITHUB_TOKEN}@github.com/$REPO"
        else
            TARGET_REPO="https://github.com/$REPO"
        fi
    else
        TARGET_REPO="$REPO"
    fi

    docker run --rm \
        -v "$TEMP_GRAPH_DIR:/app/graph_db" \
        -v "$TEMP_VECTOR_DIR:/app/vector_db" \
        -e GDSKG_GRAPH_DB_DIR=/app/graph_db \
        -e GDSKG_VECTOR_DB_DIR=/app/vector_db \
        -e GITHUB_TOKEN="$GITHUB_TOKEN" \
        gdskg:latest \
        build --repository "$TARGET_REPO" --graph /app/graph_db
done

# 6. Create custom Dockerfile and build
echo ""
echo -e "${YELLOW}Step 6: Packaging Image${NC}"

BUILD_CONTEXT=$(mktemp -d)
mkdir -p "$BUILD_CONTEXT/graph_db"
mkdir -p "$BUILD_CONTEXT/vector_db"

# Copy the data into the build context
cp -r "$TEMP_GRAPH_DIR/." "$BUILD_CONTEXT/graph_db/"
cp -r "$TEMP_VECTOR_DIR/." "$BUILD_CONTEXT/vector_db/"

cat <<EOF > "$BUILD_CONTEXT/Dockerfile"
FROM gdskg:latest

# Copy pre-built databases
COPY graph_db/ /app/graph_db/
COPY vector_db/ /app/vector_db/

# Ensure correct permissions
RUN chown -R root:root /app/vector_db /app/graph_db

# Override environment variables to point to bundled data
ENV GDSKG_VECTOR_DB_DIR=/app/vector_db
ENV GDSKG_GRAPH_DB_DIR=/app/graph_db
ENV GDSKG_MODEL_DIR=/root/.gdskg/models

# Default command - specifically configured for serving in Docker
ENTRYPOINT ["python", "main.py", "serve"]
CMD ["--transport", "stdio"]
EOF

docker build -t "$IMAGE_NAME" "$BUILD_CONTEXT"

# 7. Cleanup
echo ""
echo -e "${YELLOW}Step 7: Cleanup${NC}"
rm -rf "$TEMP_GRAPH_DIR" "$TEMP_VECTOR_DIR" "$BUILD_CONTEXT"

echo -e "${GREEN}===================================================${NC}"
echo -e "${GREEN}   Success! Image '$IMAGE_NAME' is ready.   ${NC}"
echo -e "${GREEN}===================================================${NC}"
echo ""
echo "To use this image in your mcp_config.json:"
echo ""
echo "{"
echo "  \"mcpServers\": {"
echo "    \"gdskg-custom\": {"
echo "      \"command\": \"docker\","
echo "      \"args\": [\"run\", \"-i\", \"--rm\", \"$IMAGE_NAME\"]"
echo "    }"
echo "  }"
echo "}"
echo ""
