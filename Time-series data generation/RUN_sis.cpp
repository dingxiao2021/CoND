#include <iostream>
#include <vector>
#include <deque>
#include "mex.h" 
#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <sstream>
using namespace std;
//#define N	100
#define iniP 0.5  //initial partition of I individuals
//#define infP 0.1    // probability of getting virus
//#define recP 0.5  // probability of recover from I
//~ #define T	50000 // run x round add the initial round T = x+1

//each round randomly choose one from neighbors to do the communication
int N;
int T; // run x round add the initial round T = x+1
float infP;
float recP;
// int diver=0;

int **nbhmatrix;
int **adjmatrix;
int *degree;
int *nowstate;
int *tempstate;
float *beta;
FILE *fp,*fp1,*fp2, *fp3;
void ini_pointer()
{
	adjmatrix=(int**)malloc(N*sizeof(int*));
	nbhmatrix=(int**)malloc(N*sizeof(int*));
	for(int i=0;i<N;i++)
	{
		adjmatrix[i]=(int*)malloc(N*sizeof(int));
		nbhmatrix[i]=(int*)malloc(100*sizeof(int));
	}
	nowstate=(int*)malloc(N*sizeof(int));
	tempstate=(int*)malloc(N*sizeof(int));
	degree=(int*)malloc(N*sizeof(int));
	beta=(float*)malloc(N*sizeof(float));
}
void free_pointer()
{
	free(nowstate);
	free(tempstate);
	free(degree);
	free(beta);
	for(int i=0;i<N;i++)
	{
		free(adjmatrix[i]);
		free(nbhmatrix[i]);
	}
	free(adjmatrix);
	free(nbhmatrix);
}
void initialize()
{
	int i;
	double temp;
	for(i=0; i<N; i++)
	{
		tempstate[i] = 0;
		temp = (double)rand() / RAND_MAX;
// 		if(diver==1)
// 			beta[i] = infP + temp * 0.2;
// 		else
// 			beta[i] = infP + 0.1;
		if(temp < iniP)
			nowstate[i] = 1;
		else
			nowstate[i] = 0;
	}

}

void input()
{
	int m,n;
	int i;
	for(i=0;i<N;i++)
	{
		degree[i] = 0;
	}
	fp=fopen("nb.txt","r");
	while(fscanf(fp,"%d %d",&m,&n)==2)
	{
		//printf("%d %d\n",m,n);
		nbhmatrix[m][degree[m]] = n;
		nbhmatrix[n][degree[n]] = m;
		degree[m] ++;
		degree[n] ++;
	}
	fclose(fp);
}


int communication()
{
	int i,j,tload,sum;
	double temp; // tload is the temp receiving load from a neighbor
	int neighbor;

	for(i=0; i<N; i++)
	{ // for each node tranverse its neighbors

		tload=0;
		if(nowstate[i] < 1)
		{//for S node may get virus
			for(j=0;j<degree[i];j++)
			{
				temp = (double)rand() / RAND_MAX;
				if(temp < infP)
				{ // get virus from neighbors
					neighbor = nbhmatrix[i][j];
					tload += nowstate[neighbor];//tload should be state of S or I  marked as 0 and 1,then Range shows its own state now,thus Range = nowstate[i][t-1]
				}
			}
			//nowstate[i][t]=min(1,tempstate[i][t];
			if(tload>0)
			{
				tempstate[i]=1;
			}
			else
			{
				tempstate[i]=0;
			}
		}
		else
		{//for I node won't get virus but recover
			temp = (double)rand() / RAND_MAX;
			if(temp< recP)
			{// I individuals have a chance to recovery
				tempstate[i]=0;
			}
			else
			{
				tempstate[i]=1;
			}
		}
	} // node loop over
	sum=0;
	for(i=0;i<N;i++)
	{
		nowstate[i]=tempstate[i];
		sum+=nowstate[i];
	}
	int Node=N;
	temp=(float)sum/(float)Node;
	//printf("%f\n",temp);
	if(temp<0.0001)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

void output()
{//output the timeseries state of each node
	int i;
	for(i=0; i<N; i++)
	{// node loop
		fprintf(fp,"%d ",(int)nowstate[i]);
	}
	fprintf(fp,"\n");
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
infP=mxGetScalar(prhs[0]); 
recP=mxGetScalar(prhs[1]);
N=mxGetScalar(prhs[2]); 
T=mxGetScalar(prhs[3]);

    time_t  t;
	srand((unsigned)time(&t));
	int timeT=0,flag=1;
    ini_pointer();
	input();
	initialize();
	fp = fopen("strategy.txt","w"); // write A matrices
	while(flag&&(timeT<T))
	{
		if(timeT%5000 == 0)
        {
            cout<<timeT<<endl;
        }
		output();
		flag=communication();
		timeT++;
	}
	fclose(fp);
	//free_pointer();
	cout<<timeT<<endl;
    
    
}



