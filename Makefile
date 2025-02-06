environment:
	conda env create -f codebase/environment.yml -p comptics

activate:
	# conda activate /home/michael/Github/comptics/conda_environment
	conda activate comptics

generate:
	python3.10 codebase/gen_random.py

train:
	sh codebase/run_scripts/train_fno3d.sh