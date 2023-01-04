#include<stdio.h>
#include<time.h>
#include<rfftw_threads.h>
#include<math.h>
#include"ns2d.h"

//function to intialize particle positions
void init_part(struct particle *part_main)
{
	extern int part_num;
	int i, j, part_i, index=part_num/tau_num;
	
	srand(time(NULL));
	
	for(j=0;j<tau_num;j++)
	for(i=0;i<index;i++)
	{
		part_i=j*index+i;
		part_main[part_i].x_pos=2.0*M_PI*rand()/RAND_MAX;
		part_main[part_i].y_pos=2.0*M_PI*rand()/RAND_MAX;
		part_main[part_i].x_vel=0;
		part_main[part_i].y_vel=0;
		part_main[part_i].tau_index=j;
		part_main[part_i].tau=tau_min+(double)part_main[part_i].tau_index*(tau_max-tau_min)/tau_num;
		part_main[part_i].radius=sqrt(vis*part_main[part_i].tau*4.5/rho_ratio);
	}
}
//function to update particle position and velocity
void update_part(double *x_vel_f, double *y_vel_f, struct particle *part_main, double *colli_freq, struct pos_grid *grid)
{
	extern int part_num;
	extern double dt;
	
	double x_vel_inter, y_vel_inter, x_pos, y_pos, x_vel, y_vel;
		
	int i;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, x_vel_inter, y_vel_inter, x_pos, y_pos, x_vel, y_vel)
	for(i=0;i<part_num;i++)
	{
		x_pos = part_main[i].x_pos;
		y_pos = part_main[i].y_pos;
		x_vel = part_main[i].x_vel;
		y_vel = part_main[i].y_vel;
		
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(x_pos, y_pos, x_vel_f);
		y_vel_inter = linear_interp(x_pos, y_pos, y_vel_f);
		
		//RK step 1
		x_pos+=dt*x_vel*0.5;
		y_pos+=dt*y_vel*0.5;
		
		x_vel=x_vel-dt*(x_vel-x_vel_inter)*0.5/part_main[i].tau;
		y_vel=y_vel-dt*(y_vel-y_vel_inter)*0.5/part_main[i].tau;

		//periodic boundary
		if(x_pos<0)
			x_pos+=2*M_PI;
		if(x_pos>=2*M_PI)
			x_pos-=2*M_PI;
		
		if(y_pos<0)
			y_pos+=2*M_PI;
		if(y_pos>=2*M_PI)
			y_pos-=2*M_PI;
			
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(x_pos, y_pos, x_vel_f);
		y_vel_inter = linear_interp(x_pos, y_pos, y_vel_f);
		
		//RK step2
		part_main[i].x_pos_next=part_main[i].x_pos+dt*x_vel;
		part_main[i].y_pos_next=part_main[i].y_pos+dt*y_vel;
		
		part_main[i].x_vel=part_main[i].x_vel-dt*(x_vel-x_vel_inter)/part_main[i].tau;
		part_main[i].y_vel=part_main[i].y_vel-dt*(y_vel-y_vel_inter)/part_main[i].tau;

		
		//periodic boundary
		if(part_main[i].x_pos_next<0)
			part_main[i].x_pos_next+=2*M_PI;
		if(part_main[i].x_pos_next>=2*M_PI)
			part_main[i].x_pos_next-=2*M_PI;
		
		if(part_main[i].y_pos_next<0)
			part_main[i].y_pos_next+=2*M_PI;
		if(part_main[i].y_pos_next>=2*M_PI)
			part_main[i].y_pos_next-=2*M_PI;
	}
	//openmp_e
	
	detect_collisions(part_main, colli_freq, grid);
	
	# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<part_num;i++)
	{
		part_main[i].x_pos=part_main[i].x_pos_next;
		part_main[i].y_pos=part_main[i].y_pos_next;
	}
	
	update_grid(part_main, grid);
}

void detect_collisions(struct particle *part_main, double *colli_freq, struct pos_grid *grid)
{
	extern double dt;
	extern int part_num;
	int i, index1, index2, index;
	double x1, y1, x2, y2, dist, radius_sum;;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, index1, index2, index, x1, y1, x2, y2, dist, radius_sum)
	for(i=0;i<grid_max;i++)
	{
		int x_index = i/grid_xmax;
		int y_index = i%grid_xmax;
				
		int grid_11 = x_index*grid_xmax + y_index;
		int grid_01 = (mod((x_index-1),grid_xmax))*grid_xmax + y_index;
		int grid_10 = (x_index)*grid_xmax + (mod((y_index - 1),grid_xmax));
		int grid_00 = (mod((x_index-1),grid_xmax))*grid_xmax + (mod((y_index - 1),grid_xmax));

		struct particle *part_one;
		struct particle *part_two;
		
		part_one=grid[grid_11].beg;
		while(part_one!=NULL)
		{
			part_two=part_one->next;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				x2=part_two->x_pos_next-part_one->x_pos_next;
				y2=part_two->y_pos_next-part_one->y_pos_next;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					part_two=part_two->next;
					continue;
				}
			
				dist = closest_dist(x1, y1, x2, y2);
			
				if(dist<=radius_sum)
				{
					index1 = part_one->tau_index;
					index2 = part_two->tau_index;
					index = index1*tau_num+index2;
					# pragma omp atomic
					colli_freq[index]++;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_00].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				x2=part_two->x_pos_next-part_one->x_pos_next;
				y2=part_two->y_pos_next-part_one->y_pos_next;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					part_two=part_two->next;
					continue;
				}
			
				dist = closest_dist(x1, y1, x2, y2);
			
				if(dist<=radius_sum)
				{
					index1 = part_one->tau_index;
					index2 = part_two->tau_index;
					index = index1*tau_num+index2;
					# pragma omp atomic
					colli_freq[index]++;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_10].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_01].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				x2=part_two->x_pos_next-part_one->x_pos_next;
				y2=part_two->y_pos_next-part_one->y_pos_next;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					part_two=part_two->next;
					continue;
				}
			
				dist = closest_dist(x1, y1, x2, y2);
			
				if(dist<=radius_sum)
				{
					index1 = part_one->tau_index;
					index2 = part_two->tau_index;
					index = index1*tau_num+index2;
					# pragma omp atomic
					colli_freq[index]++;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_10].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				x2=part_two->x_pos_next-part_one->x_pos_next;
				y2=part_two->y_pos_next-part_one->y_pos_next;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					part_two=part_two->next;
					continue;
				}
			
				dist = closest_dist(x1, y1, x2, y2);
			
				if(dist<=radius_sum)
				{
					index1 = part_one->tau_index;
					index2 = part_two->tau_index;
					index = index1*tau_num+index2;
					# pragma omp atomic
					colli_freq[index]++;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
		
		part_one=grid[grid_01].beg;
		while(part_one!=NULL)
		{
			part_two=grid[grid_11].beg;
			
			while(part_two!=NULL)
			{
				x1=part_two->x_pos-part_one->x_pos;
				y1=part_two->y_pos-part_one->y_pos;
				x2=part_two->x_pos_next-part_one->x_pos_next;
				y2=part_two->y_pos_next-part_one->y_pos_next;
				radius_sum=part_two->radius+part_one->radius;
			
				if((x1*x1 + y1*y1) < radius_sum * radius_sum)
				{
					part_two=part_two->next;
					continue;
				}
			
				dist = closest_dist(x1, y1, x2, y2);
			
				if(dist<=radius_sum)
				{
					index1 = part_one->tau_index;
					index2 = part_two->tau_index;
					index = index1*tau_num+index2;
					# pragma omp atomic
					colli_freq[index]++;
				}
				part_two=part_two->next;
			}
			part_one=part_one->next;
		}
	}
	//openmp_e
}

double closest_dist(double x1, double y1, double x2, double y2)
{
	double slope_perp;
	double x_perp, y_perp;

	if(x1>M_PI) x1-=2*M_PI;
	if(y1>M_PI) y1-=2*M_PI;
	if(x2>M_PI) x2-=2*M_PI;
	if(y2>M_PI) y2-=2*M_PI;
	
	slope_perp = -(x2-x1)/(y2-y1);
	
	x_perp = (x1 + slope_perp*y1)/(1.0 + slope_perp * slope_perp);
	y_perp = slope_perp * x_perp;
	
	if((x1<x_perp&&x_perp<x2)||(x2<x_perp&&x_perp<x1))
		return sqrt(x_perp*x_perp+y_perp*y_perp);
		
	if((x1*x1+y1*y1)>(x2*x2+y2*y2)) return sqrt(x2*x2+y2*y2);
	else return sqrt(x1*x1+y1*y1);
}

void update_grid(struct particle *part_main, struct pos_grid *grid)
{
	int i;
	double delta_x=2*M_PI/grid_xmax, delta_y=2*M_PI/grid_xmax;
	
	//killing grid
	# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<grid_max;i++)
	{
		grid[i].beg=NULL;
		grid[i].end=NULL;
	}
	
	//openmp_s	//err check
	//# pragma omp parallel for schedule(static) private(i)
	for(i=0;i<part_num;i++)
	{
		int x_index, y_index, index;
		
		x_index=part_main[i].x_pos/delta_x;
		y_index=part_main[i].y_pos/delta_y;
		
		index=x_index*grid_xmax+y_index;
		
		if(grid[index].beg==NULL)
		{
			grid[index].beg=&part_main[i];
			grid[index].end=grid[index].beg;
			grid[index].end->next=NULL;
		}
		else
		{
			grid[index].end->next=&part_main[i];
			grid[index].end=grid[index].end->next;
			grid[index].end->next=NULL;
		}		
	}
	//openmp_e
}

double linear_interp(double x_pos, double y_pos, double *vel)
{
	extern int sys_size;
	int x_index, y_index, index_00, index_01, index_10, index_11;
	double vel_inter, vel_temp1, vel_temp2, x_index_val, y_index_val;
	
	x_index = x_pos*sys_size/(2*M_PI);
	y_index = y_pos*sys_size/(2*M_PI);
	
	x_index_val = x_index*(2*M_PI)/sys_size;
	y_index_val = y_index*(2*M_PI)/sys_size;
	
	index_00 = x_index*(sys_size+2) + y_index;
	index_01 = x_index*(sys_size+2) + y_index + 1;
	index_10 = (x_index+1)*(sys_size+2) + y_index;
	index_11 = (x_index+1)*(sys_size+2) + y_index +1;
	
	if(x_index==sys_size-1)
	{
		index_10 = y_index;
		index_11 = y_index+1;
	}
	
	if(y_index==sys_size-1)
	{
		index_01 = x_index*(sys_size+2);
		index_11 = (x_index+1)*(sys_size+2);
	}
	
	if(x_index==sys_size-1&&y_index==sys_size-1)
		index_11 = 0;
		
	vel_temp1 = vel[index_00] + (vel[index_10]-vel[index_00])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	vel_temp2 = vel[index_01] + (vel[index_11]-vel[index_01])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	
	vel_inter = vel_temp1 + (vel_temp2-vel_temp1)*(y_pos-y_index_val)/((2*M_PI)/sys_size);
	
	return vel_inter;
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}
