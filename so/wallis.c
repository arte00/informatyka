// LINUX
// Calculating pi with threads and without ( Wallis'  formula)
// sum from 1 to n (2n)(2n)/(2n-1)(2n+1) = pi/2
// clock doesnt work correctly since it works different on linux (calculates time spent in all threads)
// but overall program works well

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>

double wallis = 1;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

struct data{
  int size;
  int start;
};

void * thread(void * thread_arg){
  
  struct data* thread_data;
  thread_data = (struct data*) thread_arg;
  
  pthread_t thread_id = pthread_self();
  int thread_start = thread_data->start;
  int thread_size = thread_data->size;
  int thread_stop = thread_start + thread_size;
  double result = 1;
  
  for(int i=thread_start; i < thread_stop; i++){
    result *= (( 2.0*i ) * ( 2.0*i )) / (( 2.0*i - 1 ) * ( 2.0*i + 1 ));
  }
  
  printf("Thread #%lxd size=%d, start=%d\n", thread_id, thread_size, thread_start );

  pthread_mutex_lock(&mutex);
  wallis *= result;
  pthread_mutex_unlock(&mutex);
  
  double* return_value = malloc(sizeof(double));
  *return_value = result;
  return return_value;
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

  
  int MAX_NUMBER = atoi(argv[1]);
  int THREADS = atoi(argv[2]);

  struct data data_array[THREADS];
  
  pthread_t threads[THREADS];
  int rc;
  int normal_size = MAX_NUMBER / THREADS;

  time_t begin1 = clock();
  
  for(int i=0; i < THREADS; i++ ){
    if( i != THREADS - 1 ){
      data_array[i].start = i * normal_size + 1;
      data_array[i].size = normal_size;
    } else {
      data_array[i].size = MAX_NUMBER - normal_size * (THREADS - 1);
      data_array[i].start = MAX_NUMBER - data_array[i].size + 1;
    }
    rc = pthread_create( &threads[i], NULL, thread, (void*) &data_array[i]);
    if( rc ){
      printf("ERROR! Return code from pthread_create(): %d\n", rc);
      exit(-1);
    }
  }
  
  for( int i=THREADS-1; i >= 0; i-- ){
    double* status;
    rc = pthread_join(threads[i], (void**) &status);
    printf("Thread #%lxd prod= %.20f\n", threads[i], *status);
    if( rc ){
      printf("ERROR! Return code from pthread_join(): %d\n", rc);
      exit(-1);
    }
    free(status);
  }

  time_t end1 = clock();
  
  double time1 = (double) (end1 - begin1) / CLOCKS_PER_SEC;
  
  printf("w/Threads: PI = %.20f time = %.5f\n", wallis*2.0, time1);
  pthread_mutex_destroy(&mutex);

  time_t begin2 = clock();
  
  double wallis_no_threads = 1;
  for( int i=1; i < MAX_NUMBER; i++ ){
    wallis_no_threads *= (( 2.0*i ) * ( 2.0*i )) / (( 2.0*i - 1 ) * ( 2.0*i + 1 ));
  }

  time_t end2 = clock();
  
  double time2 = (double) (end2 - begin2) / CLOCKS_PER_SEC;
  
  printf("wo/Threads: PI = %.20f time = %.5f\n", wallis_no_threads*2.0, time2);
  
}
