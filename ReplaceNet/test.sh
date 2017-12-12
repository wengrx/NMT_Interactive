#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu2,floatX=float32

cd $PBS_O_WORKDIR

python ${HOME}/NMT_Interactive/ReplaceNet/translate.py \
	${HOME}/NMT_Interactive/ReplaceNet/models/model_bidir_1.6m_replace.npz  \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_Interactive/Bidirection/result/transmt03_bidir_1.lctok \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_Interactive/ReplaceNet/result/transmt03_replace

