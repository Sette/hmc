import os
import argparse
from datetime import datetime as dt

from b2w.black_magic.visao.train.training import run

import tensorflow as tf


# Set python level verbosity
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

# Set C++ Graph Execution level verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(tf.compat.v1.logging.DEBUG)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--job-name", help="Nome do job de treinamento")
    parser.add_argument("--epochs", help="Número de épocas de treinamento",type=int)
    parser.add_argument("--log-steps", help="Log steps",type=int)
    parser.add_argument("--buffer-size", help="Buffer size",type=int)
    parser.add_argument("--batch-size", help="Tamanho do batch",type=int)
    parser.add_argument("--base-path", help="Caminho do job criado pela DAG")
    parser.add_argument("--train-path", help="Caminho do dataset de treinamento")
    parser.add_argument("--validation-path", help="Caminho do dataset de validação")
    parser.add_argument("--test-path", help="Caminho do dataset de teste")
    parser.add_argument("--labels-path", help="Diretório das labels")
    parser.add_argument("--metadata-path", help="Diretório do metadata do pre-processamento")
    parser.add_argument("--model", help="Argumento com o tipo do modelo")
    parser.add_argument("--job-dir")
    args = parser.parse_args()

    time_start = dt.utcnow()
    print("[Visão treinamento] Experiment started at {}".format( time_start.strftime("%H:%M:%S")))
    print(".......................................")
    time_start = dt.utcnow()
    print("[{}] Experiment started at {}".format(args.job_name, time_start.strftime("%H:%M:%S")))
    print(".......................................")
    print(args)

    run(args)

    time_end = dt.utcnow()
    time_elapsed = time_end - time_start
    print(".......................................")
    print("[{}] Experiment finished at {} / elapsed time {}s".format(args.job_name, time_end.strftime("%H:%M:%S"), time_elapsed.total_seconds()))


if __name__ == '__main__':
    main()