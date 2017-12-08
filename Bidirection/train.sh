#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=168:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q GpuQ

export THEANO_FLAGS=device=gpu1,floatX=float32

cd $PBS_O_WORKDIR
python ${HOME}/NMT_Interactive/Bidirection/train_nmt.py \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.tok \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.tok \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_Data/CH-EN/reference/MT03/ref0 \
    ${HOME}/NMT_Interactive/Bidirection/models/model_base_1.6m_old.npz \
    ${HOME}/NMT_Interactive/Bidirection/models/model_bidir_1.6m.npz