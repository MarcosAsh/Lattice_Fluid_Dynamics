.PHONY: help sim web deploy-modal deploy-web test fmt clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-15s %s\n", $$1, $$2}'

sim: ## Build the C simulation
	cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
	cmake --build build -j$$(nproc)

web: ## Install deps and start the Next.js dev server
	cd website && npm install && npm run dev

deploy-modal: ## Deploy the Modal GPU worker
	cd simulation && modal deploy modal_worker.py

deploy-web: ## Deploy the website to Vercel
	cd website && npx vercel --prod

test: sim ## Run the simulation, Python and website tests
	ctest --test-dir build --output-on-failure
	uv run pytest
	cd website && npm ci && npm run lint && npm run build && npm test

fmt: ## Format C, Python and TS
	git ls-files '*.c' '*.h' '*.cpp' '*.hpp' ':!simulation/lib/glad' | xargs clang-format -i
	uv run ruff check --fix . && uv run ruff format .
	cd website && npx eslint --fix src/

clean: ## Remove build artifacts
	rm -rf build
	rm -rf website/.next website/node_modules
