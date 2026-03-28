REPO ?= us-east4-docker.pkg.dev/akurata-offsite/util/qwen-lora-trainer
TAG  ?= 2026.3.27-4
IMG  ?= $(REPO):$(TAG)

.PHONY: build
build:
	docker build -t $(IMG) .
	docker push $(IMG)