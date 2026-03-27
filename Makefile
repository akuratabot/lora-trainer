
IMG ?= us-east4-docker.pkg.dev/akurata-offsite/util/qwen-lora-trainer:2026.3.27-3
.PHONY: build
build:
	docker build -t $(IMG) .
	docker push $(IMG)