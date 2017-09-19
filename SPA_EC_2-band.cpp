#include <iostream>
#include <utility>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

typedef complex<double> dcomp;

struct a_struct{
	const double a = 1.;
	const double v2 = -2.8;
	const double kT = 8.617342857e-5*300/13.6058;
	dcomp E;
	const double x = 0.;
	const double z = 0.;
	const double t_12 = 0.;
	const double t_21 = conj(t_12);
	const double S = 0.;
	int s;
	double P;	//period of J against spacer thickness
	const double t = 0.5;
//assuming t_11 = t_12, x = z = 0, a = 1, simple cubic 2 band model see worksheet 'SPA check 2-band simple.mw'
	const double A = 4.*(t_12*t_21 - t*t);
	const double B = A*A;

};

Matrix2cd adlayer(Matrix2cd &OM, Matrix2cd &T, Matrix2cd &Greens, double N)
{
	Matrix4cd X,O,Gamma, LHmob;
	Matrix2cd D = OM*T.inverse();
	Matrix2cd B = T.inverse();
	Matrix2cd C = T.adjoint();
	X <<	0,	0,	B(0,0),		B(0,1),
		0,	0,	B(1,0),		B(1,1),
		-C(0,0),-C(0,1),D(0,0),		D(0,1),
		-C(1,0),-C(1,1),D(1,0),		D(1,1);
	ComplexEigenSolver<Matrix4cd> ces;
	ces.compute(X);
	O = ces.eigenvectors();
	Gamma = ces.eigenvalues().asDiagonal();
	LHmob = O*(Gamma.array().pow(N)).matrix()*O.inverse();

	Matrix2cd a = LHmob.topLeftCorner(2,2);
	Matrix2cd b = LHmob.topRightCorner(2,2);
	Matrix2cd c = LHmob.bottomLeftCorner(2,2);
	Matrix2cd d = LHmob.bottomRightCorner(2,2);
	Matrix2cd G = (a*Greens + b)*((c*Greens + d).inverse());
	return G;
}

Matrix2cd greens(Matrix2cd &OM, Matrix2cd &T)
{
	Matrix2cd D = OM*T.inverse();
	Matrix2cd B = T.inverse();
	Matrix2cd C = T.adjoint();
	Matrix4cd X,O;
	X <<	0,	0,	B(0,0),		B(0,1),
		0,	0,	B(1,0),		B(1,1),
		-C(0,0),-C(0,1),D(0,0),		D(0,1),
		-C(1,0),-C(1,1),D(1,0),		D(1,1);
	ComplexEigenSolver<Matrix4cd> ces;
	ces.compute(X);
	O = ces.eigenvectors();
	Matrix2cd b = O.topRightCorner(2,2);
	Matrix2cd d = O.bottomRightCorner(2,2);
	Matrix2cd GR = b*d.inverse();
	return GR;
}

dcomp Rspace(double N, a_struct &params) {
//...F(0)|NM(n)|F(theta)...
	/* const double t_11 = 0.5; */
	double t_11 = params.t;
	double t_22 = t_11;
	const double v1 = -1.6295;
	/* const double v2 = -2.8; */
	const double v3 = -1.6295;
	const double nab = -0.6795;
	double F = cos(params.x*params.a)+cos(params.z*params.a);
	Matrix2cd Tsplit;
	Tsplit << (t_11+t_22)*F+sqrt(((t_11-t_22)*(t_11-t_22)+4.*params.t_12*params.t_21)*F*F+params.S*params.S+2.*params.S*(params.t_12+params.t_21)*F), 0,
			0,	(t_11+t_22)*F-sqrt(((t_11-t_22)*(t_11-t_22)+4.*params.t_12*params.t_21)*F*F+params.S*params.S+2.*params.S*(params.t_12+params.t_21)*F);

	Matrix2cd T((Matrix2cd() << t_11,params.t_12,params.t_21,t_22).finished());
	static const Matrix2cd V2((Matrix2cd() << params.v2,params.S,params.S,params.v2).finished());
	static const Matrix2cd V3_u((Matrix2cd() << v3+nab,params.S,params.S,v3+nab).finished());
	static const Matrix2cd V3_u_AF((Matrix2cd() << v3+nab,-params.S,-params.S,v3+nab).finished());
	static const Matrix2cd V3_d((Matrix2cd() << v3-nab,params.S,params.S,v3-nab).finished());
	static const Matrix2cd V3_d_AF((Matrix2cd() << v3-nab,-params.S,-params.S,v3-nab).finished());
	static const Matrix2cd I((Matrix2cd() << 1.,0.,0.,1.).finished());
	static const Matrix2cd V1_u((Matrix2cd() << v1+nab,params.S,params.S,v1+nab).finished());
	static const Matrix2cd V1_d((Matrix2cd() << v1-nab,params.S,params.S,v1-nab).finished());
//initialise surface Green's function using mobius transformation for RHS majority greens function
	Matrix2cd OMV3u = params.E*I-V3_u-Tsplit;
	Matrix2cd GR_u = greens(OMV3u,T);

	Matrix2cd OMV3uAF = params.E*I-V3_u_AF-Tsplit;
	Matrix2cd GR_u_AF = greens(OMV3uAF, T);

//initialise surface Green's function using mobius transformation for RHS minority greens function
	Matrix2cd OMV3d=params.E*I-V3_d-Tsplit;
	Matrix2cd GR_d = greens(OMV3d,T);

	Matrix2cd OMV3dAF = params.E*I-V3_d_AF-Tsplit;
	Matrix2cd GR_d_AF = greens(OMV3dAF, T);

//initialise surface Green's function using mobius transformation for LHS majority greens function
	Matrix2cd OMV1u=params.E*I-V1_u-Tsplit;
	Matrix2cd GL_u = greens(OMV1u,T);

//initialise surface Green's function using mobius transformation for LHS minority greens function
	Matrix2cd OMV1d=params.E*I-V1_d-Tsplit;
	Matrix2cd GL_d = greens(OMV1d,T);

	Matrix2cd OM = params.E*I - V2 - 2.*T*F;

	Matrix2cd GN_u = adlayer(OM, T, GL_u, N);
	Matrix2cd GN_d = adlayer(OM, T, GL_d, N);

	dcomp Fsigma;
	Matrix2cd Rsigma_u_FM, Rsigma_d_FM, Rsigma_u_AF, Rsigma_d_AF;

	Rsigma_u_FM = (I-GR_u*T.adjoint()*GN_u*T);
	Rsigma_d_FM = (I-GR_d*T.adjoint()*GN_d*T);
	/* Rsigma_u_AF = (I-GR_d_AF*T.adjoint()*GL_u*T); */
	Rsigma_u_AF = (I-GR_d*T.adjoint()*GN_u*T);
	/* Rsigma_d_AF = (I-GR_u_AF*T.adjoint()*GL_d*T); */
	Rsigma_d_AF = (I-GR_u*T.adjoint()*GN_d*T);
	Fsigma = (1./M_PI)*log((Rsigma_u_FM*Rsigma_d_FM*Rsigma_u_AF.inverse()*Rsigma_d_AF.inverse()).determinant());
	
	dcomp i;
	i = -1.;
	i = sqrt(i);
	dcomp factor = exp(-i*2.*M_PI*static_cast<dcomp>(params.s*N)/params.P);
	/* cout<<"factor = "<<factor<<"N, s = "<<N<<" "<<params.s<<endl; */
	Fsigma *= factor;

	return (1/params.P)*Fsigma;
}

dcomp integration(a_struct &params)
{
	double start = 18.;
	double end = start + params.P;
	int n = 150;
	double h = (end - start)/(2.*n);
	dcomp result = 0.;
	double N = start;
	result += Rspace(N, params);
	N = end;
	result += Rspace(N, params);
	#pragma omp parallel
	{
	#pragma omp for nowait reduction(+:result)
	for (int j = 1; j <= n; j++){
		N = start + (2*j - 1.)*h;	
		result += 4.*Rspace(N, params);
		if (j != n){
			N = start + 2*j*h;
			result += 2.*Rspace(N, params);
		}
	}}
	result *= h/3.;
	/* cout<<"s = "<<params.s<<", c_s = "<<result<<endl; */
	return result;
}

double dsi_dE(double psi, a_struct &params)
{
	
	const double delta = 1e-1;
	params.E -= delta;
	dcomp C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	dcomp D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 
				+ params.S*params.t*(params.t_12 + params.t_21)) + 
				params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	dcomp y_1 = acos((C + D)/params.A);
	dcomp y_2 = acos((C - D)/params.A);
	dcomp phi = 2.*M_PI - y_1 - y_2;
	/* params.P = (M_PI/real(y_1))*(M_PI/real(y_2)); */
	/* params.P = M_PI/real(phi); */
	params.P = M_PI/real(y_1);
	dcomp delta_E_n = integration(params);
	params.P = M_PI/real(y_2);
	delta_E_n *= integration(params);
	params.E += delta;

	params.E += delta;
	C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 
				+ params.S*params.t*(params.t_12 + params.t_21)) + 
				params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	y_1 = acos((C + D)/params.A);
	y_2 = acos((C - D)/params.A);
	phi = 2.*M_PI - y_1 - y_2;
	/* params.P = M_PI/real(phi); */
	/* params.P = (M_PI/real(y_1))*(M_PI/real(y_2)); */
	params.P = M_PI/real(y_1);
	dcomp delta_E_p = integration(params);
	params.P = M_PI/real(y_2);
	delta_E_p *= integration(params);
	params.E -= delta;	

	params.E -= 1.2*delta;
	C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 
				+ params.S*params.t*(params.t_12 + params.t_21)) + 
				params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	y_1 = acos((C + D)/params.A);
	y_2 = acos((C - D)/params.A);
	phi = 2.*M_PI - y_1 - y_2;
	/* params.P = M_PI/real(phi); */
	/* params.P = (M_PI/real(y_1))*(M_PI/real(y_2)); */
	params.P = M_PI/real(y_1);
	dcomp delta_E_2n = integration(params);
	params.P = M_PI/real(y_2);
	delta_E_2n *= integration(params);
	params.E += 1.2*delta;

	params.E += 1.2*delta;
	C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 
				+ params.S*params.t*(params.t_12 + params.t_21)) + 
				params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	y_1 = acos((C + D)/params.A);
	y_2 = acos((C - D)/params.A);
	phi = 2.*M_PI - y_1 - y_2;
	/* params.P = (M_PI/real(y_1))*(M_PI/real(y_2)); */
	/* params.P = M_PI/real(phi); */
	params.P = M_PI/real(y_1);
	dcomp delta_E_2p = integration(params);
	params.P = M_PI/real(y_2);
	delta_E_2p *= integration(params);
	params.E -= 1.2*delta;

	double psi_n = atan2(imag(delta_E_n),real(delta_E_n));

	double psi_p = atan2(imag(delta_E_p),real(delta_E_p));

	double psi_2n = atan2(imag(delta_E_2n),real(delta_E_2n));

	double psi_2p = atan2(imag(delta_E_2p),real(delta_E_2p));
/* cout<<psi<<'\t'<<psi_n<<'\t'<<psi_p<<endl; */
	double dsi_dE = (psi_p - psi_n)/(2*delta);
	double test = (psi_2p - psi_2n)/(2.4*delta);
	double abs_err = abs(abs(dsi_dE/test)-1);
	/* cout<<1/sinh(dsi_dE)<<endl; */

	if ( abs_err > 0.05)
		cout<<"caution: for s = "<<params.s<<", dpsi/dE has absolute error of "<<abs_err<<endl;
	if ( abs(abs(2.*psi/(psi_p + psi_n))-1) > 0.05)
		cout<<"caution: for s = "<<params.s<<", psi is not on the gradient line"<<endl;

	/* cout<<params.s<<" "<<dsi_dE<<endl; */
	return dsi_dE;
}
Matrix2cd hessian(a_struct &params)
{
	//caution, results here are specific to fixed parameters as described above, calculated in said worksheet
	dcomp C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	dcomp D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 + 
				params.S*params.t*(params.t_12 + params.t_21)) + 
				params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	dcomp el_11 = -params.A/sqrt(params.B - (C - D)*(C - D)) - params.A/sqrt(params.B - (C + D)*(C + D));
	double el_12 = 0.;
	dcomp el_22 = el_11;
	Matrix2cd Hes;
	Hes << el_11,	el_12,
	    	el_12,	el_22;
	/* cout<<el_11<<'\t'<<el_12<<'\n'<<el_12<<'\t'<<el_22<<'\n'<<endl; */
	return Hes;
}
		
int main() 
{
	//FOR C_S SIGN OF IMAGINARY COMPONENT IS DEPENDENT ON SIGN OF S. THIS SHOULD NOT BE THE CASE, NOT THE CASE FOR REAL COMPONENT OR ANDREY'S REAL OR IMAGINARY.
//why doesn't dsi_dE contribute to spincurrent?
	a_struct params;
	double N = 50;

	dcomp i;
	i = -1.;
	i = sqrt(i);
	params.E = 0.0 + i*1e-10;

	dcomp C = 2.*params.A + 2.*params.t*(params.E - params.v2) + params.S*(params.t_12 + params.t_21);
	dcomp D = sqrt(4.*(params.E - params.v2)*((params.E - params.v2)*params.t_12*params.t_21 + params.S*params.t*(params.t_12 + params.t_21)) + params.S*params.S*((params.t_12 - params.t_21)*(params.t_12 - params.t_21) + 4.*params.t*params.t));
	dcomp F = 2.*(params.S*params.t*(params.t_12 + params.t_21) + 2.*(params.E - params.v2)*params.t_12*params.t_21);
	//dy/dE = -1/sqrt(-(E+v2-F)^2+1
	//or more obviously.. since E = u + 2t(cos(x)+cos(y)+cos(z), dE/dy = -2tsin(y)..
	/* double phi = (1./params.a)*acos((real(params.E)-params.v2)/(2.*params.t)-2); */
	dcomp y_1 = acos((C + D)/params.A);
	dcomp y_2 = acos((C - D)/params.A);
	dcomp phi = 2.*M_PI - y_1 - y_2;
	dcomp first_term, second_term;
	if (abs(F) == 0){
		first_term = (2.*params.t)/sqrt(params.B - (C - D)*(C - D));
		second_term = (2.*params.t)/sqrt(params.B - (C + D)*(C + D));
	}
	else
	{
		first_term = (2.*params.t - F/D)/sqrt(params.B - (C - D)*(C - D));
		second_term = (2.*params.t + F/D)/sqrt(params.B - (C + D)*(C + D));
	}
	dcomp dphi_dE = first_term + second_term;

	// see SPA check for J(N).mw... d derived from this worksheet by solving determinant 
	// of Hessian matrix, only valid for Ef and v2 as given in worksheet
	Matrix2cd Hes = hessian(params);
	double det_hes = sqrt(abs(Hes(0,0)*Hes(1,1)-Hes(0,1)*Hes(1,0)));
	/* cout<<det_hes<<endl; */
	dcomp tau;
	/* cout<<Hes.trace()<<endl; */
	ComplexEigenSolver<Matrix2cd> es;
	es.compute(Hes);
	VectorXcd eigenvals = es.eigenvalues();

	if ((real(eigenvals(0)) < 0) && (real(eigenvals(1)) < 0))
		tau = -i;
	if ((real(eigenvals(0)) > 0) && (real(eigenvals(1)) > 0))
		tau = i;
	if (((real(eigenvals(0)) > 0) && (real(eigenvals(1)) < 0)) || ((real(eigenvals(0)) < 0) && (real(eigenvals(1)) > 0)))
		tau = 1.;
	int fourier_start = -5;
	int fourier_end = 0;
	int size = (fourier_end-fourier_start);
	VectorXcd C_s;
	VectorXd dPsi_dE;
	C_s.resize(size);
	dPsi_dE.resize(size);
	
	dcomp c_s;

	int index;
	for (params.s = fourier_start; params.s != fourier_end; params.s++){
		index = params.s - fourier_start;
		if (params.s == 0){
			C_s[index] = 0;
			dPsi_dE[index] = 0;
			continue;}
		/* params.P = (M_PI/real(y_1))*(M_PI/real(y_2)); */
		/* params.P = M_PI/real(phi); */
		params.P = M_PI/real(y_1);
		c_s = integration(params);
		params.P = M_PI/real(y_2);
		c_s *= integration(params);
		double psi = atan2(imag(c_s),real(c_s));
		double dpsi_dE = dsi_dE(psi, params);
		C_s[index] = c_s;
		dPsi_dE[index] = dpsi_dE;
	}

	double spincurrent, result;

	cout<<"Name the data file\n";
	string Mydata;
	getline(cin, Mydata);
	ofstream Myfile;	
	Mydata += ".txt";
	Myfile.open( Mydata.c_str(),ios::trunc );
	Myfile<<"N , J(N)"<<endl;

	/* double j = 20.; */

	for (double j = 1; j<=N; j = j + 0.1){
		spincurrent = 0.;
		// s here is the summation of Fourier terms
		
		for (params.s = fourier_start; params.s != fourier_end; params.s++){
			if (params.s == 0)
				continue;
			index = params.s - fourier_start;
			result = real(tau*C_s[index]*exp(i*static_cast<double>(j*params.s)*phi)
				/(params.s*det_hes*sinh(M_PI*params.kT*(j*params.s*dphi_dE + dPsi_dE[index]))));
			spincurrent += result;
			/* cout<<"s: "<<params.s<<"\tc_s: "<<C_s[index]<<endl; */
		}
		Myfile<< j <<" , "<< -(params.kT/(2.*j))*spincurrent << endl;

	}

	cout<<"finished!"<<endl;


return 0;
}
