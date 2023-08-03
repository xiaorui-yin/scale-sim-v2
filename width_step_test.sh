#!/bin/zsh

for ofm_sz in {64..1024..64}
do
  sed "10s/.*/OfmapSramSzkB: ${ofm_sz}/" configs/ne005.cfg > configs/temp.cfg

  python3 scalesim/scale.py -t topologies/user/test_width_step.csv -c configs/temp.cfg | tee ./width_step_test/width_step_$ofm_sz.log

  rm configs/temp.cfg
done
