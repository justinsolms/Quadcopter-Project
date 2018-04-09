floyd init justinsolms/ddpg_quadcopter 

# justinsolms/projects/ddpg_quadcopter
floyd run \
  --env tensorflow-1.2 \
  --gpu \
  --message "RL-QuadCopter-2" \
    "python ddpg_quadcopter.py \
      --dropout=0.5 \
    "
