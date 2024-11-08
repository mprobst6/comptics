activate:
	conda activate /home/michael/Github/comptics/conda_environment

generate:
	python3.10 codebase/gen_random.py

train:
	sh run_scripts/train_fno3d.sh