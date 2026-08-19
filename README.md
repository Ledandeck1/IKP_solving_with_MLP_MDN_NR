Olá, o repositório contém os arquivos utilizados para gerar um experimento de 60 redes neurais artificiais de 4 camadas ocultas, variando entre MLP, MLP com Mixture Density Network como output e loss function,
e MLP com método numérico de Newton-Raphson.
Para gerar o dataset, composto por ângulos distribuídos uniformemente aleatórios dentro do range de atuação das juntas de um manipulador de 6-DOF, basta rodar o código em generate_dataset.py que faz a leitura
da cinemática direta de forwardkinematics.py com os parâmetros do braço robótico alvo, e da cinemática direta resultante desses ângulos gerados, formando um dataset de 500000 amostras XYZ theta1 theta2 theta3
theta4 theta5 theta6.
separate_dataset.py serve para gerar o arquivo que alimenta as ANNs e normalizar os dados.
Em seguida temos um script de train e test para um mlp, a fim de validar a ideia inicial do projeto apenas para um simples MLP.
o arquivo train_z_mdn.py contém as basesd de treino com a MDN, que é chamado dentro de run_experiments. Há também o arquivo de test para o mdn apenas, onde é testado cada mixture da mdn.
run_experiments é o script de controle experimental que faz os 60 treinamentos.
newplot.py é o script utilizado para gerar dados e gráficos visando colocá-los no paper da matéria de tópicos em IA.

Os pesos treinados, o dataset utilizado, imagens de treino e validação, não foi possível anexar no github, então esse é o caminho para reproduzir o experimento realizado.

Etapas que não deram o resultado final tratam-se de variar tamanho da arquitetura, número de mixtures, e outros hiperparâmetros, onde os valores obtidos pouco ou nada diferenciou do resultado com a arquitetura definida.
os arquivos não anexados são por exceder o tamanho permitido para envio.
