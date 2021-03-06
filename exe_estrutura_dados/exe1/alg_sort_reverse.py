#!/usr/bin/python3
# -*- coding: utf-8 -*-

DEFAULT_OUT = "out_alg_sort_selection.txt"
DEFAULT_SEED = None

DEFAULT_N_START = 1
DEFAULT_N_STOP = 10
DEFAULT_N_STEP = 1
DEFAULT_TRIALS = 3

from subprocess import Popen, PIPE
from time import sleep, time
from multiprocessing import Process
import shlex
import json

import sys
import os
import argparse
import logging
import subprocess

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import timeit


class Node:

    def __init__(self, data):
        self.data = data
        self.proximo = None


class ListaEncadeada:

    def __init__(self):
        self.cabeca = None

    def reverter(self):
        if self.cabeca is None or self.cabeca.proximo is None:
            return

        anterior = None
        atual = self.cabeca

        while atual:
            proximo_elemento = atual.proximo
            atual.proximo = anterior
            anterior = atual
            atual = proximo_elemento

        self.cabeca = anterior

    def push(self, data):
        novo_nodo = Node(data)
        novo_nodo.proximo = self.cabeca
        self.cabeca = novo_nodo

    def print_list(self):
        atual = self.cabeca
        l1 = []
        while atual:
            l1.append(atual.data)
            atual = atual.proximo
        return l1


def sort_reverse(lista):
	"""
	Implementação do Sort Reverse
	:param lista: qualquer ordem
	:return: lista reversa
	"""
	'''n = len(lista)-1
	m = len(lista)
	for i in range(m):
		aux = lista[n]
		lista[n] = i
		lista[i] = aux
		n -= 1'''
	cabeca = ListaEncadeada()
	for i in lista:
		cabeca.push(i)
	cabeca.reverter()
	lista = cabeca.print_list()
	return lista


def main():
	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Naive TPS')
	help_msg = "arquivo de saída.  Padrão:{}".format(DEFAULT_OUT)
	parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUT, type=str)

	help_msg = "semente aleatória. Padrão:{}".format(DEFAULT_SEED)
	parser.add_argument("--seed", "-s", help=help_msg, default=DEFAULT_SEED, type=int)

	help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_STOP)
	parser.add_argument("--nstop", "-n", help=help_msg, default=DEFAULT_N_STOP, type=int)

	help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
	parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

	help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

	help_msg = "tentativas.        Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--trials", "-t", help=help_msg, default=DEFAULT_TRIALS, type=int)

	# Lê argumentos from da linha de comando
	args = parser.parse_args()


	trials = args.trials
	f = open(args.out, "w")
	f.write("#Reverse sort\n")
	f.write("#n time_s_avg time_s_std (for {} trials)\n".format(trials))
	m = 100
	np.random.seed(args.seed)
	for n in range(args.nstart, args.nstop+1, args.nstep): #range(1, 100):
		resultados = [0 for i in range(trials)]
		tempos = [0 for i in range(trials)]
		for trial in range(trials):
			print("\n-------")
			print("n: {} trial: {}".format(n, trial+1))
			entrada = np.random.randint(0, n, n)
			print("Entrada: {}".format(entrada))
			tempo_inicio = timeit.default_timer()
			resultados[trial] = sort_reverse(entrada)
			tempo_fim = timeit.default_timer()
			tempos[trial] = tempo_fim - tempo_inicio
			print("Saída: {}".format(resultados[trial]))
			print('Tempo: {} s'.format(tempos[trial]))
			print("")

		tempos_avg = np.average(tempos)  # calcula média
		tempos_std = np.std(a=tempos, ddof=False)  # ddof=calcula desvio padrao de uma amostra?


		f.write("{} {} {}\n".format(n, tempos_avg, tempos_std))
	f.close()


if __name__ == '__main__':
	sys.exit(main())
