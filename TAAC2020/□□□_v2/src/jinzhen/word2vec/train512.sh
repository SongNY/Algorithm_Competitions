
path_cache=../../../cache/
path_corpus=$path_cache/corpus/
path_w2v=$path_cache/word2vec/
mkdir -p path_w2v


./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win4_min1 -size 512 -sample 8e-4 -window 4 -negative 15 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win4_min1 -size 512 -sample 8e-5 -window 4 -negative 15 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win4_min1 -size 512 -sample 8e-5 -window 4 -negative 15 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1


./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win8_min1 -size 512 -sample 8e-4 -window 8 -negative 10 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win8_min1 -size 512 -sample 8e-5 -window 8 -negative 10 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win8_min1 -size 512 -sample 8e-5 -window 8 -negative 10 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1


./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win16_min1 -size 512 -sample 9e-4 -window 16 -negative 8 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win16_min1 -size 512 -sample 9e-4 -window 16 -negative 8 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win16_min1 -size 512 -sample 9e-4 -window 16 -negative 8 -cbow 0 -iter 5 -threads 24 -output-mode 1 -min-count 1 -debug 1



./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win32_min1 -size 512 -sample 1e-3 -window 32 -negative 6 -cbow 0 -iter 4 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win32_min1 -size 512 -sample 1e-4 -window 32 -negative 6 -cbow 0 -iter 4 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win32_min1 -size 512 -sample 1e-4 -window 32 -negative 6 -cbow 0 -iter 4 -threads 24 -output-mode 1 -min-count 1 -debug 1



./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win64_min1 -size 512 -sample 1e-3 -window 64 -negative 4 -cbow 0 -iter 3 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win64_min1 -size 512 -sample 1e-4 -window 64 -negative 4 -cbow 0 -iter 3 -threads 24 -output-mode 1 -min-count 1 -debug 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win64_min1 -size 512 -sample 1e-4 -window 64 -negative 4 -cbow 0 -iter 3 -threads 24 -output-mode 1 -min-count 1 -debug 1


./word2vec -train $path_corpus/ader.txt -output $path_w2v/ader512/win128_min1 -size 512 -sample 7e-4 -window 128 -negative 1 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1 -fix-window 1
./word2vec -train $path_corpus/item.txt -output $path_w2v/item512/win128_min1 -size 512 -sample 8e-5 -window 128 -negative 1 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1 -fix-window 1
./word2vec -train $path_corpus/adid.txt -output $path_w2v/adid512/win128_min1 -size 512 -sample 8e-5 -window 128 -negative 1 -cbow 0 -iter 7 -threads 24 -output-mode 1 -min-count 1 -debug 1 -fix-window 1

