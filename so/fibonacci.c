// This program prints n-th fibonacci nuber in console ( and pids and return codes of programs it created )
// program [n]
// n = n-th number of fibonacci sequence
// needs to be updated, so it always starts, not only when you name it correctly

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <Windows.h>
#include <stdio.h>

int main(int argc, char** argv)
{
    if (argc != 2){
        fprintf(stderr, "Nieprawidlowa ilosc argumentow\n");
        exit(1);
    }

    char* argument = argv[1];
    int i = 0;
    if (argument[0] == '-'){
        i = 1;
    }

    for(; argument[i] != 0; i++){
        if((!isdigit(argument[i]))){
       fprintf(stderr, "Argument nie jest liczba calkowita\n");
       exit(2);
       }
    }

    if (atoi(argv[1]) < 1 || atoi(argv[1]) > 13){
    fprintf(stderr, "Argument nie znajduje sie w przedziale <1;13>\n");
    exit(3);
    }

    if (atoi(argv[1]) == 1 || atoi(argv[1]) == 2){
    exit(1);
    }


    int current_arg = atoi(argv[1]);
    STARTUPINFO si;
    PROCESS_INFORMATION pi[2];

    memset(&si, 0, sizeof(si));
    memset(&pi, 0, sizeof(pi));
    si.cb = sizeof(si);

    for(int i=0; i < 2; i++){
        char argline[50];
        sprintf(argline, "46586.so.lab09.exe %d", current_arg-i-1);
        if(! CreateProcessA(NULL, argline, NULL, NULL, 0, 0, NULL, NULL, &si, &pi[i])){
            int err = GetLastError();
            printf("CreateProcessA %d failed (%d).\n", i+1, err );
        }
    }

    DWORD exit_code1, exit_code2;

    WaitForSingleObject(pi[1].hProcess, INFINITE);
    WaitForSingleObject(pi[0].hProcess, INFINITE);

    GetExitCodeProcess(pi[0].hProcess, &exit_code1);
    GetExitCodeProcess(pi[1].hProcess, &exit_code2);

    printf("%d\t%d\t%d\t%ld\n", (int) GetCurrentProcessId(), pi[0].dwProcessId, current_arg-1, exit_code1);
    printf("%d\t%d\t%d\t%ld\n", (int) GetCurrentProcessId(), pi[1].dwProcessId, current_arg-2, exit_code2);
    int exit_status_sum = exit_code1 + exit_code2;
    printf("%d\t\t\t%d\n\n", (int) GetCurrentProcessId(),  exit_status_sum);

    for( int i=0; i < 2; i++){
        CloseHandle(pi[i].hProcess);
        CloseHandle(pi[i].hThread);
    }

    exit(exit_status_sum);
}
