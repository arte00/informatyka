// Calculating pi (Leibnitz's formula) with threads and without ( time comparisson )
// Sum of i=0, i < n -1^2 / 2n + 1 = pi / 4
// function [n] [t]
// n = max number in sum, t = number of threads
// hubi


#include <stdlib.h>
#include <stdio.h>
#include <windows.h>
#include <ctype.h>
#include <time.h>

double powerMinusOne(int n){
    return (n % 2) ? -1.0 : 1.0;
}

typedef struct{
    int start;
    int size;
    double retval;
}DATA;

double leibnitz = 0;

HANDLE mutex;

DWORD WINAPI thread(LPVOID data){
    DATA *dat = (DATA*) data;
    DWORD thread_id = GetCurrentThreadId();
    int thread_start = dat->start;
    int thread_size = dat->size;
    int thread_stop = thread_start + thread_size;
    double result = 0;
    printf("Thread #%lxd size=%d, start=%d\n", thread_id, thread_size, thread_start );
    for(int i=thread_start; i < thread_stop; i++){
        result += powerMinusOne(i) / (2.0 * i + 1.0);
    }
    dat->retval = result;

    WaitForSingleObject(mutex, INFINITE);
    leibnitz += result;
    ReleaseMutex(mutex);
}

int main(int argc, char **argv){
    if( argc != 3){
    fprintf(stderr, "Invalid number of arguments\n");
    exit(1);
  }

  for( int i = 1; i <= 2; i++){

    char* argument = argv[i];
    int i = 0;
    for( ; argument[i] != 0; i++ ){
      if( !(isdigit(argument[i])) ){
	fprintf(stderr, "Invailid argument\n");
	exit(1);
      }
    }
  }
  
  if( atoi(argv[1]) < 1 || atoi(argv[1]) > 1000000000 ){
    fprintf(stderr, "Argument (n)  not in  <1;1000000000>\n");
    exit(1);
  }
  
  if( atoi(argv[2]) < 1 || atoi(argv[2]) > 100 ){
    fprintf(stderr, "Argument (w) not in <1;100>\n");
    exit(1);
  }

  mutex = CreateMutex(NULL, FALSE, NULL);

  
  const int MAX_NUMBER = atoi(argv[1]);
  const int THREADS = atoi(argv[2]);

  HANDLE threads[THREADS];
  DWORD threadIds[THREADS];
  DATA data[THREADS];

  int normal_size = MAX_NUMBER / THREADS;

  time_t begin1 = clock();

  for( int i=0; i < THREADS; i++){
    if( i != THREADS - 1 ){
      data[i].start = i * normal_size;
      data[i].size = normal_size;
    } else {
      data[i].size = MAX_NUMBER - normal_size * (THREADS - 1);
      data[i].start = MAX_NUMBER - data[i].size;
    }
    threads[i] = CreateThread(NULL, 0, thread, data + i, 0, threadIds + i);
  }

  for(int i=0; i < THREADS; i++){
      WaitForSingleObject(threads[i], INFINITE);
      printf("Thread #%lxd completed sum=%.20f\n", threadIds[i], data[i].retval);
      CloseHandle(threads[i]);
  }

  time_t end1 = clock();
  double time1 = (double) (end1 - begin1) / CLOCKS_PER_SEC;
  
  printf("w/Threads: PI = %.20f time = %.5f\n", leibnitz * 4.0, time1);

  time_t begin2 = clock();
  
  double leibnitz_no_threads = 0;
  for( int i=0; i < MAX_NUMBER; i++ ){
    leibnitz_no_threads += powerMinusOne(i) / (2.0 * i + 1.0);
  }

  time_t end2 = clock();
  
  double time2 = (double) (end2 - begin2) / CLOCKS_PER_SEC;
  
  printf("wo/Threads: PI = %.20f time = %.5f\n", leibnitz_no_threads * 4.0, time2);

  return 0;
}
