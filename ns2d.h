#ifndef ns2d_h
#define ns2d_h

extern double tolerence;
extern int sys_size, part_num, nthreads, tau_num; 
extern const int grid_max, grid_xmax;
extern double dt, vis, mu, tau_min, tau_max, rho_ratio;

struct particle
{
	int part_index;
	double x_pos, x_pos_next;
	double y_pos, y_pos_next;
	double x_vel;
	double y_vel;
	double radius;
	double tau;
	int tau_index;
	struct particle *next;
};

struct pos_grid
{
	struct particle *beg;
	struct particle *end;
};
	
void init_den_state(int *den_state);

void init_omega(fftw_complex *omega_ft, int *den_state);

void find_vel_ft(fftw_complex *omega_ft, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft);

void find_e_spectra(fftw_complex *x_vel, fftw_complex *y_vel, double *e_spectra);

void gen_force(double famp, int kf);

void find_jacobian_ft(double *omega, double *x_vel, double *y_vel);

double find_energy(double *e_spectra);

double find_epsilon(fftw_complex *omega_ft);

void solve_rk2 (fftw_complex *omega_ft, fftw_complex *omega_t, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft, struct particle *part_main, double *colli_freq, struct pos_grid *grid);

void init_part(struct particle *part_main);

void update_part(double *x_vel, double *y_vel, struct particle *part_main, double *colli_freq, struct pos_grid *grid);

void detect_collisions(struct particle *part_main, double *colli_freq, struct pos_grid *grid);

void update_grid(struct particle *part_main, struct pos_grid *grid);

double closest_dist(double x1, double y1, double x2, double y2);

int mod(int a, int b);

double linear_interp(double x_pos, double y_pos, double *vel);

#endif
