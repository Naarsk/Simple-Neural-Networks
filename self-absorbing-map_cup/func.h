float randf(){                                                  /*random float */
    return (float)rand()/(float)(RAND_MAX);
    }
    
void init(float **w, int r, int c){                             /*weight random inizialization*/
    int     i,j                     ;
    for(i = 0; i < r; i++){  
        for(j = 0; j < c; j++){                                 
            w[i][j]=randf()*2-1     ;                         
            }
        }
    return                          ;
    }
    
float dist(float *x,float *w,int l){                             /*2-distance*/
    float   ds=0                    ;
    int     i                       ;
    for(i=0;i<l;i++){           
        ds+=(x[i]-w[i])*(x[i]-w[i]) ;
        }                            
    return ds                       ;                           
    }    
    
int scl(int *k, int a, int *jv){                                /*scalarizes a vectorial index*/
    int js=0,c=1,i                  ;
    for(i = 0; i < a; i++){
        js=js+c*jv[i]               ;
        c=c*k[i]                    ;
        }
    return js                       ;
    }

void retr(int js, int *k, int *jv, int a){                      /*retrieves components for a given index*/ 
    int   c=1,i                     ;                           /*converting from scalar to vector representation*/
    for(i = 0; i < a; i++){
        c=c*k[i]                    ;
        }
    if(js>c){
        printf("error: point outside the box");    
        }
    i = a                           ;
    while(i>0) {
        i=i-1                       ;  
        c=c/k[i]                    ;
        jv[i]=js/c                  ;  
        js=js-jv[i]*c               ;
        }
    return                          ;
    }

int win(float **w, float *p, int *k, int n, int a){                       /*decides which neuron in the winner*/
    float   d,dmin                  ;
    int     c=1,i,j,jmin            ;
    for(i = 0; i < a; i++){
        c=c*k[i]                    ;
        }
    dmin=10000                      ;
    for(j = 0; j < c; j++){
            d=dist(p,w[j],n)        ;
            if (d<dmin){
                dmin=d              ; 
                jmin=j              ;
                }
            }
    return jmin                     ;
    }

float f(float s,int* j,int* jmin, int a){                      /*gaussian weighting function*/
    int c=0,i                               ;
    for(i = 0; i < a; i++){
        c=c+(j[i]-jmin[i])*(j[i]-jmin[i])   ;                   /*int 2-distance*/
        }
    return exp(-c/(2*s*s))                  ;   
    }


void hebb(float **w, float *p, int* k, float h, float s, int* jmin, int n, int a){         /*adjourning neurons*/
    int c=1,i,j                             ;
    int jv[a]                               ;

    for(i = 0; i < a; i++){
        c=c*k[i]                            ;
        }
    for(j = 0; j < c; j++){
        retr(j,k,jv,a)                      ;
        for(i = 0; i < n; i++){
            w[j][i]=w[j][i]+h*f(s,jv,jmin,a)*(p[i]-w[j][i])   ;
            }
        }
    return ;
    }

