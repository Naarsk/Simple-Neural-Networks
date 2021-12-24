#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "nrutil.h"
#include "nrutil.c"
#include "var.h"
#include "func.h"


#define UMAX  10000	                /*max no. of cycles */ 	
#define PMAX  10000                 /*max no. of error evaluations*/
#define EMIN  1/10000.0             /*min error on training set*/
#define GMIN  7.8/100.0             /*min error on testing set*/
#define H   0.01                    /*step size*/

#define N0  54                      /*no. of inputs*/
#define N1  4                       /*no. of neurons in the 1st hidden layer*/
#define N2  3                       /*no. of neurons in the 2nd hidden layer*/
#define N3  1                       /*no. of outputs*/
#define M   100                     /*no. of exemples used for training*/
#define MT  70                      /*no. of exemples used for testing*/    


int main(){  
    int   i0,i1,i2,i3,j,k,u ;        
    int   p = 0             ;
    float r = 0             ;
    float d = 0             ;
    float e = 10            ;
    float g = 10            ;
    
    float y0[N0+1]          ;       /* +1 for threshold*/
    float y1[N1+1]          ;
    float y2[N2+1]          ;
    float y3[N3+1]          ;

    float **w0              ;       /*input layer weights*/
    float **w1              ;       /*1st hidden layer weights*/
    float **w2              ;       /*2nd hidden layer weights*/

    w0=matrix(0,N1,0,N0+1)  ;
    w1=matrix(0,N2,0,N1+1)  ;
    w2=matrix(0,N3,0,N2+1)  ;     
    
    srand(time(NULL))       ;
    init(w0,N1,N0+1)        ;
    init(w1,N2,N1+1)        ;
    init(w2,N3,N2+1)        ;
    

    float **x               ;       
    float **xt              ;       
    
    x =matrix(0,M,0,N0+1)   ;       /*+1 for results*/
    xt=matrix(0,MT,0,N0+1)  ;

    for(i0 = 0; i0 < N0+1; i0++){
        for(j = 0; j < M; j++){
            x[j][i0]=dat[j][i0];
            }
        for(j = 0; j < MT; j++){
            xt[j][i0]=datt[j][i0];
            }
        }
        
    norm(x,M,N0+1)          ;
    norm(xt,MT,N0+1)        ;
    
    printf("initial relative error values (probably around 0.5 since the output is binary):\n")   ;
    printf("E=%f \t G= %f \n",error(x, M, N0, N1, N2, N3, w0, w1, w2),error(xt, MT, N0, N1, N2, N3, w0, w1, w2));
    
    
    FILE * outfile                         ;
    outfile = fopen("plot.dat","w")        ;
	
    
    /*NETWORK TRAINING*/

    for (u = 0; u < UMAX; u++){
        
        /*FORWARD*/
        
        srand(time(NULL)+37*u)  ;
        k = rand() % M          ;          /* random int between 0 and m */
        r=x[k][N0]              ;

        for (i0 = 0; i0 < N0; i0++){
            y0[i0]=x[k][i0]     ;
            }
        y0[N0]=1.0              ;
        
        layer(y0,w0,N0+1,y1,N1) ;        
        layer(y1,w1,N1+1,y2,N2) ;      
        layer(y2,w2,N2+1,y3,N3) ;

        d = r-y3[0]             ;
            
        /*BACKWARD*/
        
        for (i3 = 0; i3<N3; i3 ++){
            for (i2 = 0; i2<N2; i2 ++){
                w2[i3][i2] = w2[i3][i2] + H * d * agg(y2,w2[i3],N2+1) * y2[i2] ;                                                                   
                for (i1 = 0; i1<N1; i1 ++){
                    w1[i2][i1] = w1[i2][i1] + H * d * agg(y2,w2[i3],N2+1) * w2[i3][i2] * agg(y1,w1[i2],N1+1) * y1[i1]  ;     
                    for (i0 = 0; i0<N0; i0 ++){
                        w0[i1][i0] = w0[i1][i0] + H * d * agg(y2,w2[i3],N2+1) * w2[i3][i2] * agg(y1,w1[i2],N1+1) * w1[i2][i1] * agg(y0,w0[i1],N0+1) * y0[i0] ; 
                        }
                    }
                }
            }
            

        /* ERROR FUNCTION PLOT*/
        
        if (u%(UMAX/PMAX)==0){
        
            e=error(x, M, N0, N1, N2, N3, w0, w1, w2)      ;       /*learning set*/
            g=error(xt, MT, N0, N1, N2, N3, w0, w1, w2)    ;       /*generalization*/
            
            fprintf(outfile,"%d \t %f \t %f \n",p,e,g)     ;
            p=p+1                                          ;
            }
        
        /*EXIT CLAUSE*/
        
        if (e<EMIN){
            printf("training completed successfully (E < %f) \t after (less than) %d cycles \n", EMIN, p*(UMAX/PMAX));
            printf("E=%f \t G= %f \n",error(x, M, N0, N1, N2, N3, w0, w1, w2),error(xt, MT, N0, N1, N2, N3, w0, w1, w2));

            break;
            }
        else if (g<GMIN){
            printf("training completed successfully (G < %f) \t after (less than) %d cycles \n", GMIN, p*(UMAX/PMAX));
            printf("E=%f \t G= %f \n",error(x, M, N0, N1, N2, N3, w0, w1, w2),error(xt, MT, N0, N1, N2, N3, w0, w1, w2));

            break;
            }
        else if (u==UMAX-1){
            printf("number of cycles exceeded %d. \t E = %f \t G = %f \n", UMAX, e, g);
            }
        }
        fclose(outfile);

    return 0 ;
    }
