floyd init justinsolms/ddpg_quadcopter

# justinsolms/projects/ddpg_quadcopter
floyd run \
  --env tensorflow-1.2 \
  --gpu \
  --message "RL-QuadCopter-2" \
    "python ddpg_quadcopter.py \
    --verbose=3 \
    --dropout=0.2 --learn_r=0.0001 \
    --theta=2.0 --sigma=0.1 --action-init-var=0.001 \
    --hidden_units_1=512 --hidden_units_2=128 \
    --nb_steps=2000000 --memory=2000000
    "
