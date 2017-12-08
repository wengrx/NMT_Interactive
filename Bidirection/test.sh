#!/usr/bin/env bash
#/bin/bash
#PBS -l nodes=1:ppn=054
#PBS -l walltime=054:00:00
#PBS -N session05_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu0,floatX=float32

cd $PBS_O_WORKDIR

python ${HOME}/NMT_Interactive/Bidirection/translate.py \
	${HOME}/NMT_Interactive/Bidirection/models/model_bidir_1.6m.npz  \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT05.cn.dev \
	${HOME}/NMT_Data/CH-EN/reference/MT05 \
	${HOME}/NMT_Interactive/Bidirection/result/transmt05_bidir