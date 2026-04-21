#################### PACKAGE ACTIONS ###################

install_package:
	@pip uninstall -y my_shelves || :
	@pip install -e .

pylint:
	find . -iname "*.py" -not -path "./tests/*" | xargs -n1 -I {}  pylint --output-format=colorized {}; true

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

run_api:
	uvicorn src.my_shelves.api.api:app --reload

build_container_local:
	docker build --tag=${DOCKER_IMAGE_NAME}:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 ${DOCKER_IMAGE_NAME}:dev

build_for_production:
	docker build \
		--platform linux/amd64 \
		-t ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${DOCKER_IMAGE_NAME}:prod \
		-f src/my_shelves/api/Dockerfile \
		.

push_image_production:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${DOCKER_IMAGE_NAME}:prod

deploy_to_cloud_run:
	gcloud run deploy \
		--image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${ARTIFACTSREPO}/${DOCKER_IMAGE_NAME}:prod \
		--memory ${MEMORY} \
		--region ${GCP_REGION}

deploy: build_for_production push_image_production deploy_to_cloud_run


deploy_web_app:
	gcloud run deploy --region ${GCP_REGION} --source src/app my-shelves
