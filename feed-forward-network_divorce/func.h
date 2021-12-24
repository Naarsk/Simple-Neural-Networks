float randf(){                                                  /*random float */
    return (float)rand()/(float)(RAND_MAX);
    }
    
float f( float x){                                              /*transfer function*/
    return tanh(x)                  ;                           
    }                                                           
    
float df(float x){
    return 1-tanh(x)*tanh(x)        ;
    }      

float dot(float *x,float *w,int l){                             /*dot product*/
    float   dp=0                    ;
    int     i                       ;
    for(i=0;i<=l;i++){
        dp+=x[i]*w[i]               ;
        }                       
    return dp                       ;                           
    }

float agg(float *x, float *w, int l){                           /*derivative of transfer function*/
    return df(dot(x,w,l))           ;
    }
   
void layer(float *x, float **w, int l, float *y, int n){        /*layer output*/
    int i                           ;
    for(i = 0; i < n; i++){                                 
        y[i] = f(dot(x,w[i],l))     ;                           /*first index for rows, second for columns*/
    y[n]=1                          ;                           /*threshold*/
            }
    return                          ;
    }

void init(float **w, int r, int c){                             /*weight random inizialization*/
    int     i,j                     ;
    for(i = 0; i < r; i++){  
        for(j = 0; j < c; j++){                                 
            w[i][j]=randf()/10;                         
            }
        }
    return                          ;
    }

void norm(float **x, int r, int c){                             /*normalization of exemples in [-1,1]*/
    float   xmax[c],xmin[c]         ;
    int     i,j                     ;
    for(j = 0; j < c; j++){ 
        xmax[j]=-100                ;
        xmin[j]=100                 ;                        
        for(i = 0; i < r; i++){
            if (x[i][j]>xmax[j]){
                xmax[j]=x[i][j]; 
                }
            if (x[i][j]<xmin[j]){
                xmin[j]=x[i][j]; 
                }
            }
        }
    for(j = 0; j < c; j++){                                 
        for(i = 0; i < r; i++){
            x[i][j]=(2*x[i][j]-xmax[j]+xmin[j])/(xmax[j]-xmin[j]);
            }
        }
    return ;
    }

float error(float** x, int m, int n0, int n1, int n2, int n3, float ** w0, float ** w1, float ** w2){
    float   *y0,*y1,*y2,*y3,*y4,r,d,er=0;                        /*relative error over a dataset*/
    int     i0,j                        ;
    y0=(float*)malloc((n0+n1+n2+n3+4)*sizeof(float));
    y1=y0+n0+1;
    y2=y1+n1+1;
    y3=y2+n2+1;    
    if(y0==NULL){
        printf("\n puntatore nullo \n");
        }
    else{
        for(j = 0; j < m; j++){
            for (i0 = 0; i0 < n0; i0++){
                y0[i0]=x[j][i0]     ;
                }
            y0[n0]=1                ;
            r=x[j][n0]              ;
            layer(y0,w0,n0+1,y1,n1) ;        
            layer(y1,w1,n1+1,y2,n2) ;      
            layer(y2,w2,n2+1,y3,n3) ;
            d = r-y3[0]             ;
            er= er + d*d/2          ;
            }
        }    
    free(y0)                		;	 
    return  er/(1.0*m)             	;
    }
