#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Rodrigo Mansilha
# Universidade Federal do Pampa (Unipampa)
# Programa de Pós-Graduação em Eng. de Software (PPGES)
# Bacharelado em: Ciência da Camputação, Eng. de Software, Eng. de Telecomunicações

# Algoritmos
# Laboratório 3: avaliação analítica



try:
	import sys
	import os
	import argparse
	import logging
	import subprocess
	import shlex
	import logging
	from abc import ABC, abstractmethod

	from scipy.special import factorial
	import math
	import numpy as np
	import matplotlib.pyplot as plt
	import scipy.optimize as opt
	import matplotlib.colors as colors
	import matplotlib.cm as cmx


except ImportError as error:
	print(error)
	print()
	print("1. (optional) Setup a virtual environment: ")
	print("  python3 -m venv ~/Python3env/algoritmos ")
	print("  source ~/Python3env/algoritmos/bin/activate ")
	print()
	print("2. Install requirements:")
	print("  pip3 install --upgrade pip")
	print("  pip3 install -r requirements.txt ")
	print()
	sys.exit(-1)

DEFAULT_N_START = 1
DEFAULT_N_STOP = 100
DEFAULT_N_STEP = 1
DEFAULT_OUTPUT = None
DEFAULT_MODELOS = None # executa todos

DEFAULT_LOG_LEVEL = logging.INFO
TIME_FORMAT = '%Y-%m-%d,%H:%M:%S'

# Lista completa de mapas de cores
# https://matplotlib.org/examples/color/colormaps_reference.html
mapa_cor = plt.get_cmap('tab20')  # carrega tabela de cores conforme dicionário
mapeamento_normalizado = colors.Normalize(vmin=0, vmax=19)  # mapeamento em 20 cores
mapa_escalar = cmx.ScalarMappable(norm=mapeamento_normalizado, cmap=mapa_cor)  # lista de cores final

formatos = ['.', 'v', 'o', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h']


# https://matplotlib.org/3.3.2/api/_as_gen/matplotlib.pyplot.plot.html
# '.'	point marker
# ','	pixel marker
# 'o'	circle marker
# 'v'	triangle_down marker
# '^'	triangle_up marker
# '<'	triangle_left marker
# '>'	triangle_right marker
# '1'	tri_down marker
# '2'	tri_up marker
# '3'	tri_left marker
# '4'	tri_right marker
# 's'	square marker
# 'p'	pentagon marker
# '*'	star marker
# 'h'	hexagon1 marker
# 'H'	hexagon2 marker
# '+'	plus marker
# 'x'	x marker
# 'D'	diamond marker
# 'd'	thin_diamond marker
# '|'	vline marker
# '_'	hline marker


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyle_str = [
     ('solid', 'solid'),      # Same as (0, ()) or '-'
     ('dotted', 'dotted'),    # Same as (0, (1, 1)) or '.'
     ('dashed', 'dashed'),    # Same as '--'
     ('dashdot', 'dashdot')]  # Same as '-.'

linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

def funcao_fatorial(n, cpu):
	'''
	Aproximação fatorial
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (factorial(n) * cpu)


def funcao_quadratica(n, cpu):
	'''
	Aproximação quadrática
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (n * n * cpu)


def funcao_linear(n, cpu):
	'''
	Aproximação linear
	:param n: tamanho da instância
	:param cpu: fator de conversão para tempo de CPU
	:return: aproximação
	'''
	return (n * cpu)


def imprime_config(args):
	'''
	Mostra os argumentos recebidos e as configurações processadas
	:args: parser.parse_args
	'''
	logging.info("Argumentos:\n\t{0}\n".format(" ".join([x for x in sys.argv])))
	logging.info("Configurações:")
	for k, v in sorted(vars(args).items()):
		logging.info("\t{0}: {1}".format(k, v))
	logging.info("")


class ModeloAnalitico(ABC):

	@abstractmethod
	def __init__(self, args):
		self.id = "letraid"
		self.args = args
		self.tamanhos = range(args.nstart, args.nstop, args.nstep)
		self.linestyle = 'solid'

		# configurações de plotagem f(x)
		self.fn_legenda = "f(n)="
		self.fn_cor_rgb = mapa_escalar.to_rgba(1)
		#self.fn_formato = formatos[1]

		# configurações de plotagem upper bound g(x)
		self.gn1_legenda = "g(n)="
		self.gn1_constante = None
		self.gn1_cor_rgb = mapa_escalar.to_rgba(3)
		#self.gn1_formato = formatos[1]

		# configurações de plotagem lower bound g(x)
		self.gn2_legenda = "g(n)="
		self.gn2_constante = None
		self.gn2_cor_rgb = mapa_escalar.to_rgba(5)
		# self.gn2_formato = formatos[1]

	@abstractmethod
	def f(self, n):
		pass

	@abstractmethod
	def g(self, n, c):
		pass

	def plota_fn(self):
		valores = [self.f(n) for n in self.tamanhos]
		logging.debug("valores: {}".format(valores))
		plt.plot(self.tamanhos, valores, linestyle=self.linestyle, label=self.fn_legenda, color=self.fn_cor_rgb)

	def plota_gn(self, constante, legenda, cor):
		valores = [self.g(n, constante) for n in self.tamanhos]
		logging.debug("valores: {}".format(valores))
		plt.plot(self.tamanhos, valores, linestyle=self.linestyle, label=legenda, color=cor)

	def plota(self):
		self.plota_fn()
		if self.gn1_constante is not None:
			self.plota_gn(self.gn1_constante, self.gn1_legenda, self.gn1_cor_rgb)
		if self.gn2_constante is not None:
			self.plota_gn(self.gn2_constante, self.gn2_legenda, self.gn2_cor_rgb)


class Exemplo1(ModeloAnalitico):

	def __init__(self, args):
		super().__init__(args)
		self.id = "1"
		self.linestyle = 'solid'

		# configurações de plotagem f(x)
		self.fn_legenda = "f(n)=3n^2 −100n +6 "

		# configurações de plotagem upper bound g(x)
		self.gn1_constante = 3
		self.gn1_legenda = "g(n)=n^2, c=%d"%self.gn1_constante

		# configurações de plotagem lower bound g(x)
		self.gn2_constante = 1
		self.gn2_legenda = "g(n)=n^2, c=%d" % self.gn2_constante

	def f(self, n):
		return 3*(n*n) -100*n +6

	def g(self, n, c):
		return n*n*c


class Exemplo2(ModeloAnalitico):

	def __init__(self, args):
		super().__init__(args)
		self.id = "2"
		self.linestyle = 'dotted'

		# configurações de plotagem f(x)
		self.fn_legenda = "f(n)=3n^2 −100n +6"

		# configurações de plotagem upper bound g(x)
		self.gn1_constante = 1
		self.gn1_legenda = "g(n)=n^3, c=%d"%self.gn1_constante

	def f(self, n):
		return 3*(n*n) -100*n +6

	def g(self, n, c):
		return n*n*n*c

class Exemplo3(ModeloAnalitico):

	def __init__(self, args):
		super().__init__(args)
		self.id = "3"
		self.linestyle = ':'

		# configurações de plotagem f(x)
		self.fn_legenda = "f(n)=3n^2 −100n +6"

		# configurações de plotagem upper bound g(x)
		self.gn1_constante = 10
		self.gn1_legenda = "g(n)=n, c=%d"%self.gn1_constante

	def f(self, n):
		return 3*(n*n) -100*n +6

	def g(self, n, c):
		return n*c



def main():
	'''
	Programa principal
	:return:
	'''

	# Definição de argumentos
	parser = argparse.ArgumentParser(description='Laboratório 3')

	help_msg = "n máximo.          Padrão:{}".format(DEFAULT_N_STOP)
	parser.add_argument("--nstop", "-n", help=help_msg, default=DEFAULT_N_STOP, type=int)

	help_msg = "n mínimo.          Padrão:{}".format(DEFAULT_N_START)
	parser.add_argument("--nstart", "-a", help=help_msg, default=DEFAULT_N_START, type=int)

	help_msg = "n passo.           Padrão:{}".format(DEFAULT_N_STEP)
	parser.add_argument("--nstep", "-e", help=help_msg, default=DEFAULT_N_STEP, type=int)

	help_msg = "figura (extensão .png ou .pdf) ou nenhum para apresentar na tela.  Padrão:{}".format(DEFAULT_OUTPUT)
	parser.add_argument("--out", "-o", help=help_msg, default=DEFAULT_OUTPUT, type=str)

	help_msg = "modelos (1=exemplo1, 2=exemplo2, 3=exemplo3) ou nenhum para executar todos.  Padrão:{}".format(
		DEFAULT_MODELOS)
	parser.add_argument("--modelos", "-m", help=help_msg, default=DEFAULT_MODELOS, type=str)

	help_msg = "verbosity logging level (INFO=%d DEBUG=%d)" % (logging.INFO, logging.DEBUG)
	parser.add_argument("--verbosity", "-v", help=help_msg, default=DEFAULT_LOG_LEVEL, type=int)

	# Lê argumentos da linha de comando
	args = parser.parse_args()

	# configura o mecanismo de logging
	if args.verbosity == logging.DEBUG:
		# mostra mais detalhes
		logging.basicConfig(format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
							datefmt=TIME_FORMAT, level=args.verbosity)

	else:
		logging.basicConfig(format='%(message)s',
							datefmt=TIME_FORMAT, level=args.verbosity)

	# imprime configurações para fins de log
	imprime_config(args)

	# lista de modelos disponíveis
	modelos = [Exemplo1(args), Exemplo2(args), Exemplo3(args)]

	for m in modelos:
		if args.modelos is None or m.id in args.modelos:
			m.plota()

	# configurações gerais
	plt.legend()
	#plt.xticks(range(args.nstart, args.nstop+1, args.nstep))
	plt.title("Luiz Guilherme Fritsch")
	plt.xlabel("Tamanho da instância (n)")
	plt.ylabel("Função")

	if args.out is None:
		# mostra
		plt.show()
	else:
		# salva em png
		plt.savefig(args.out, dpi=300)


if __name__ == '__main__':
	sys.exit(main())
