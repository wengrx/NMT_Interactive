#!/usr/bin/env bash
#/bin/bash
#PBS -l nodes=1:ppn=054
#PBS -l walltime=054:00:00
#PBS -N session05_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu3,floatX=float32

cd $PBS_O_WORKDIR

python ${HOME}/NMT_Interactive/Adapting/adapting_translate.py \
	${HOME}/NMT_Interactive/Bidirection/models/model_bidir_1.6m.npz  \
	${HOME}/NMT_Data/CH-EN/train/cn.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/train/en.1.6m.pkl \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_Interactive/Bidirection/result/transmt03_bidir_1.lctok \
	${HOME}/NMT_Data/CH-EN/source/MT03.cn.dev \
	${HOME}/NMT_Interactive/Bidirection/result/transmt03_adapting_bidir
#   	${HOME}/NMT_Data/CH-EN/reference/MT03/ref0 \