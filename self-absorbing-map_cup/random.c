#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define PI    3.14159265            /*pi greco*/

#define N     5000                  /*no. of desired points*/ 	
#define M     3.14159265            /*mean value of the gaussian*/
#define S     0.5                   /*variance of the gaussian*/
#define R     1.0                   /*radius of rhe sphere*/

float randf(){                                                  /*random float */
    return (float)rand()/(float)(RAND_MAX);
    }

int main(){          
    int   i                        ;
    float v1,v2,v3,ph,th           ;
    float x[N],y[N],z[N]           ;    
    
    FILE * outfile                         ;        /*file per c*/
    outfile = fopen("var.h","w")           ;
    
    FILE * outfile2                        ;        /*file per mathematica*/
    outfile2 = fopen("poin.dat","w")       ;


    fprintf(outfile,"float dat[%i][3] = {",N)  ;


    /*GENERATION OF A GAUSSIAN DISTRO ON A SPHERE*/
    srand(time(NULL))     ;

    for (i = 0; i < N; i++){
               
        v1=randf()                  ;
        v2=randf()                  ;
        v3=pow(-2*log(v1),0.5)*sin(2*PI*v2);        /*Box-Muller transform*/
        
        ph=2*PI*randf()             ;
        th=M+v3*S                   ;
        
        x[i]=R*sin(th)*cos(ph)      ;
        y[i]=R*sin(th)*sin(ph)      ;
        z[i]=R*cos(th)              ;
           
        if(i==N-1){
            fprintf(outfile,"{%f,%f,%f}};",x[i],y[i],z[i])   ;          /*scrittura per c*/
            }
        else{
            fprintf(outfile,"{%f,%f,%f},\n",x[i],y[i],z[i])  ;
            }
                    
        fprintf(outfile2,"%f \t %f \t %f \n",x[i],y[i],z[i]) ;          /*scrittura per mathematica*/
    
        }
    
    fclose(outfile)                                    ;
    fclose(outfile2)                                   ;

    return 0;
    }
