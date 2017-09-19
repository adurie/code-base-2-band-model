#ifndef CUNNINGHAM_POINTS_ADAPTIVE_H
#define CUNNINGHAM_POINTS_ADAPTIVE_H
#include <vector>
#include <complex>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <functional>
#include <numeric>
//This header performs Cunningham points integration in x and z along the triangle (pi/a)*(pi/a)/2 (the irreducible segment).
//It is doubly adaptive as it subdivides in x and z.
//IMPORTANT actual Cunningham Special Points doesn't compute points on x-axis, instead offsets by half the step distance
//so step distance is consistent when multiplied over full zone. Either make this change, or scale points on x-axis appropriately
//x-axis scaled, as well as gamma point (0,0) as didn't like that this isn't computed using standard CP. if ((k==l) || (z==0)) 0.5*...

double Reset_tmp(const std::vector<double> &tmp){
	return 0.0;
}

std::complex<double> Reset_tmp(const std::vector<std::complex<double>> &tmp){
	std::complex<double> i = -1;
	i = std::sqrt(i);
	return 0.0*i;
}

Eigen::VectorXcd Reset_tmp(const std::vector<Eigen::VectorXcd> &integral){
	Eigen::VectorXcd tmp = Eigen::VectorXcd::Zero((integral[0]).size());
	return tmp;
}

Eigen::VectorXd Reset_tmp(const std::vector<Eigen::VectorXd> &integral){
	Eigen::VectorXd tmp = Eigen::VectorXd::Zero((integral[0]).size());
	return tmp;
}

double Accumulate(const std::vector<double> &integral){
	return std::accumulate(integral.begin(), integral.end(), 0.0);
}

std::complex<double> Accumulate(const std::vector<std::complex<double>> &integral){
	std::complex<double> i = -1;
	i = std::sqrt(i);
	return std::accumulate(integral.begin(), integral.end(), 0.*i);
}

Eigen::VectorXd Accumulate(const std::vector<Eigen::VectorXd> &integral){
	Eigen::VectorXd zero = Eigen::VectorXd::Zero((integral[0]).size());
	return std::accumulate(integral.begin(), integral.end(), zero);
}

Eigen::VectorXcd Accumulate(const std::vector<Eigen::VectorXcd> &integral){
	Eigen::VectorXcd zero = Eigen::VectorXcd::Zero((integral[0]).size());
	return std::accumulate(integral.begin(), integral.end(), zero);
}

double Abs_Result(double result){
	return std::abs(result);
}

double Abs_Result(std::complex<double> result){
	return std::abs(result);
}

double Abs_Result(const Eigen::VectorXd &result){
	return std::abs(result.sum());
}

double Abs_Result(const Eigen::VectorXcd &result){
	return std::abs(result.sum());
}

double Condition(std::complex<double> result, std::complex<double> part_result){
	double condition;
	if ((std::abs(part_result) == 0) && (std::abs(result) == 0))
		condition = 0;
	else
		condition = std::abs(std::abs(part_result)/std::abs(result)-1.);
	return condition;
}
double Condition(double result, double part_result){
	double condition;
	if ((std::abs(part_result) == 0) && (std::abs(result) == 0))
		condition = 0;
	else
		condition = std::abs(std::abs(std::abs(part_result)/(result))-1.);
	return condition;
}
double Condition(const Eigen::VectorXd &result, const Eigen::VectorXd &part_result){
	double condition;
	if ((std::abs(part_result.sum()) == 0) && (std::abs(result.sum()) == 0))
		condition = 0;
	else
		condition = std::abs(std::abs(part_result.sum())/std::abs(result.sum())-1.);
	return condition;
}
double Condition(const Eigen::VectorXcd &result, const Eigen::VectorXcd &part_result){
	double condition;
	if ((std::abs(part_result.sum()) == 0) && (std::abs(result.sum()) == 0))
		condition = 0;
	else
		condition = std::abs(std::abs(part_result.sum())/std::abs(result.sum())-1.);
	return condition;
}

template <typename func, typename... Args, typename ret>//Auxilliary recursive function in z, called after x is defined
ret aux_kspace(func&& predicate, int depth, double rel, int step_size, double scale_x, double scale_z,
	       	const std::vector<ret>& integral, int k, double x, double start, double end, double ext_start, const double a, Args&&... params){

	double z;
	double middle = (end-start)/2. + start;
	int loop_end;
	std::vector<ret> integral_left;
	std::vector<ret> integral_right;
	int side = k-1;
	int k_aux = k;
	if (ext_start > 0)// this is the special case where the triangles to integrate over are above squares. Described below.
		k_aux = 1;
	if (k == 0){//special case where subintegral is a square since after integral divides in x, RHS is composed of a
	       	//square and triangle. k == 0 is a flag to indicate this and k is actually treated as n, or step_size here.
		loop_end = 2*step_size;
		side = step_size;//ceiling here is one less than end, as end coincides with start of next square, or triangle
		k_aux = step_size + 1;
		integral_left.reserve(step_size);
		integral_right.reserve(step_size);
	}
	else {
		if (end < M_PI/a){//in this case, end coincides with start on RHS, so the end of the loop is adapted to accommodate this
			loop_end = 2*(k-1);
			integral_left.reserve(k-1);
			integral_right.reserve(k-1);
		}
		if (end == M_PI/a){
			loop_end = 2*k-1;
			integral_left.reserve(k-1);
			integral_right.reserve(k);
		}
	}
	for (int l=1; l<=loop_end; l++){//ext_start here is required only for the triangles that are
	       //above the squares. Geometrically, the triangles are calculated at x = 0, and raised to the correct level by ext_start.
		z = (end - start)*(l-1)/(2.*step_size) + ext_start + start*(k_aux-1)/(1.*step_size);
		if (l<=side){//splits into R & L
			if (l%2 == 0)
				integral_left.emplace_back((1./(2.*scale_x*step_size*scale_z*step_size))*
						std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));
			else
				integral_left.emplace_back((0.5)*integral[(l - 1)/2]);//seen here and below, attempts have been made not to recalculate degenerate 
			//points, instead 'remember' them from previous cycles.
		}
		if (l>side){
			if ((l%2 == 0) || (l > 2*integral.size())){
				if ((2*k)==l){//this point is halved to take into account the multiplication of the final result to get the full Brillouin Zone 
					integral_right.emplace_back((1./(2.*scale_x*step_size*scale_z*step_size))*0.5*
							std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
				if ((2*k)!=l){
					integral_right.emplace_back((1./(2.*scale_x*step_size*scale_z*step_size))*
							std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
			}
			else
				integral_right.emplace_back(0.5*integral[(l - 1)/2]);
		}
	}
	ret part_result = Accumulate(integral);//accumulates for all four data types that this header is overloaded for
	ret result_left = Accumulate(integral_left);
	ret result_right = Accumulate(integral_right);
	ret result = result_left + result_right;
	double condition = Condition(result, part_result);//condition overloaded for all data types this header is designed for
	double abs_result = Abs_Result(result);//abs overloaded for all data types considered here
	if ((condition > 0.7) && (abs_result > 1e-4))//if the result is too erroneous, the function keeps iterating until convergence is achieved.
		depth++;// This is because if neighbouring points are zero, the result will appear to 
			//halve each time as each recursion is scaled, until convergence.
	if (depth <=0 || condition <= rel)
		return result;
	if ((depth <= 0) && (condition > rel))
		std::cout<<"warning, for x, z = "<<x<<", "<<z<<" maximum depth reached with error "<<condition<<std::endl;
	return aux_kspace(predicate, depth-1, rel, step_size, scale_x, 2*scale_z, integral_left, k, x, start, middle, ext_start, a, params...) +
		aux_kspace(predicate, depth-1, rel, step_size, scale_x, 2*scale_z, integral_right, k, x, middle, end, ext_start, a, params...);
}

template <typename func, typename... Args, typename ret>//auxilliary recursive function in x
ret aux_aux_kspace(func&& predicate, int depth, int aux_depth, double rel, int step_size,
	       	double scale, const std::vector<ret>& integral, double start, double end, const double a, Args&&... params){

	std::vector<ret> right;
	std::vector<ret> left;
	left.reserve(step_size/2);
	if (end == M_PI/a)
		right.reserve(step_size/2 + 1);
	else
		right.reserve(step_size/2);
	int n = step_size;
	double middle = (end - start)/2. + start;
	double x, z;
	ret tmp = Reset_tmp(integral);//zero's tmp to the correct dimension for the four data types considered here
	double aux_start = 0.0;
	double aux_end = end-start;
	for (int k=1; k<=n+1; k++) {
		if ((end != M_PI/a) && (k == n + 1))//avoids duplicate calculations of LHS end and RHS start
			continue;
		x = (end - start)*(k-1)/n + start;
		tmp = Reset_tmp(integral);//as above
		if ((k-1)%2 != 0){
			aux_start = 0.0;
			aux_end = end-start;
			if (start > 0){
		  		while (aux_start < start){ //stacks squares until the diagonal z = x is hit, then builds triangle

					std::vector<ret> square_l_integral;
					square_l_integral.reserve(n+1);
					for (int l=1; l<=n+1; l++) {
						z = (aux_end - aux_start)*(l-1)/n + aux_start;
						if ((k==l) || (z==0)){
							square_l_integral.emplace_back((1./(scale*scale*n*n))*0.5*
									std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
						if (k!=l){
							square_l_integral.emplace_back((1./(scale*scale*n*n))*
									std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
					}
					tmp += aux_kspace(predicate, aux_depth, rel, n, scale/1., scale, square_l_integral, 0, x, aux_start, aux_end, 0, a, params...); 

					//here 'k' is passed as 0 to allow special case of square
					aux_start += end - start;
					aux_end += end - start;
				}
			}

			std::vector<ret> l_integral;
			l_integral.reserve(k);
			for (int l=1; l<=k; l++) {//this bit builds the triangle on the squares
				z = (aux_end - aux_start)*(l-1)/n + aux_start;
				if ((k==l) || (z==0)){
					l_integral.emplace_back((1./(scale*scale*n*n))*0.5*std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
				if (k!=l){
					l_integral.emplace_back((1./(scale*scale*n*n))*std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
			}
			if (k > 1)
				tmp += aux_kspace(predicate, aux_depth, rel, n, scale, scale, l_integral, k, x, 0., end - start, aux_start, a, params...); 
				//the strange 'end' here is to ensure dimensions of triangle are correct. Essentially 'built' on x = 0 and raised to 
				//correct position aux_start.
				
		}
		if (k <= step_size/2)//splits into R & L
		{
			if ((k-1)%2 != 0)
				left.emplace_back(tmp);
			if ((k-1)%2 == 0)
				left.emplace_back(0.5*integral[(k - 1)/2]);
		}
		if (k > step_size/2)
		{
			if ((k-1)%2 != 0)
				right.emplace_back(tmp);
			else
				right.emplace_back(0.5*integral[(k - 1)/2]);
		}
	}
			
	ret part_result = Accumulate(integral);
	ret result_left = Accumulate(left);
	ret result_right = Accumulate(right);
	ret result = result_left + result_right;
	double condition = Condition(result, part_result);
	if ((depth <= 0) && (condition > rel))
		std::cout<<"warning maximum depth reached!"<<std::endl;

	if (depth <=0 || condition <= rel)
		return result;
	ret left_ret = aux_aux_kspace(predicate, depth-1, aux_depth, rel, step_size, 2*scale, left, start, middle, a, params...);
	ret right_ret = aux_aux_kspace(predicate, depth-1, aux_depth, rel, step_size, 2*scale, right, middle, end, a, params...);
	return left_ret + right_ret;
}

template <typename func, typename... Args>
auto kspace(func&& predicate, int depth, double rel, int step_size, const double a, Args&&... params)
    -> decltype(std::forward<func>(predicate)(std::declval<double>(), std::declval<double>(), std::declval<double>(),std::forward<Args>(params)...)) {
	    typedef decltype(std::forward<func>(predicate)(std::declval<double>(), std::declval<double>(), std::declval<double>(),std::forward<Args>(params)...)) ret;

	//params is a struct passing parameters to the forwarded function 'predicate'
	//depth is the recursion cap, rel is the relative error being worked to
	//step_size is the number of points integrated over at each recursion
	//depth, rel & step_size take default values if 0 is passed in their place
	int n, Depth;
	if (step_size == 0)
		n = 768;
	else
		n = step_size;
	if (depth == 0)
		Depth = 3;
	else
		Depth = depth;
	double error;
	if (rel == 0)
		error = 1e-2;
	else
		error = rel;
	std::vector<ret> k_integral;
	k_integral.reserve(n+1);
	ret tmp;
	double A = M_PI/a;
	double x, z;

	for (int k=1; k<=n+1; k++) {
		x = A*(k-1)/n;
		std::vector<ret> l_integral;
		l_integral.reserve(k);
		for (int l=1; l<=k; l++) {
			z = A*(l-1)/n;
			if ((k==l) || (z==0)){
				l_integral.emplace_back((1./(n*n))*0.5*std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
			if (k!=l){
				l_integral.emplace_back((1./(n*n))*std::forward<func>(predicate)(x,z,a,std::forward<Args>(params)...));}
		}

		if (k==1){//bypasses null calculations since x = 0 here.
			tmp = Accumulate(l_integral);
			tmp /= 4.;
		}
		if (k>1)
			tmp = aux_kspace(predicate, Depth, error, n, 1., 1., l_integral, k, x, 0, A, 0, a, params...); 
		k_integral.emplace_back(tmp);

	}
	ret result_return = aux_aux_kspace(predicate, Depth, Depth/2, error, 2*n, 1., k_integral, 0, A, a, params...);
	return 8.*A*A*result_return;
}

#endif
