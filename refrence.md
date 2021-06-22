```bash
opencv_createsamples -info info.lst -num 5011 -w 50 -h 50 -vec positives.vec
```

```bash
opencv_traincascade -data output -vec positives.vec -bg bg.txt \
 -numPos 2000 -numNeg 1000 -numStages 2 -vec positives.vec
 ```