#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu1,floatX=float32

cd $PBS_O_WORKDIR

python ${HOME}/NMT_Interactive/L2R/translate.py \
	${HOME}/NMT_Interactive/L2R/params/model_bidir_1.6m.npz  \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT02.cn.dev \
	${HOME}/NMT_Data/CH-EN/reference/MT02 \
	${HOME}/NMT_Interactive/L2R/result/transmt02_l2r

