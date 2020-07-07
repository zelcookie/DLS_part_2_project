# Neural Machine Translation


1. Написать(стырить из домашки) Baseline – LSTM + Attn. Обучить его на максимальный скор. 

Baseline LSTM + Attn.ipynb 

2. Написать свой Трансформер для перевода. Получить на нем максимальный скор. Сохранить веса(!)

transformer.ipynb - Трансформер на слоях из torch

3. Посмотреть, как "срезать" головы с Трансформеров. Будем это делать как в статье от [Voita et al.](https://www.aclweb.org/anthology/P19-1580/).
4. Написать код для среза голов. Посмотреть, как хорошо он работает.

transformer-prune.ipynb обучение Трансформера и файнтюниг с прунингом на слоях из torch + MultiheadAttention из Annotated Transformer переделанные для прунинга на основе статьи из п.3

