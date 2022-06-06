from scipy.cluster.vq import kmeans2
import gpytorch
import torch

def add_model_args(parser):
    parser.add_argument('--model', type=str, default="svgp")
    
    
    # likelihood params
    parser.add_argument('--likelihood', type=str, default="gaussian")
    parser.add_argument('--noise-scale', type=float, default=0.05)
    
    # kernel params
    parser.add_argument('--kernel', type=str, default="ard")
    parser.add_argument('--lon-scale', type=float, default=1.0)
    parser.add_argument('--lat-scale', type=float, default=1.0)
    parser.add_argument('--time-scale', type=float, default=7.0)
    parser.add_argument("--kernel-variance", type=float, default=1.0)
    
    # variational params
    parser.add_argument("--n-inducing", type=int, default=100)
    parser.add_argument("--learn-inducing", type=bool, default=True)
    parser.add_argument("--variational-tril", type=bool, default=False)

    return parser
        
    
def get_inducing_points(data, args):
    
    n_inducing = args.n_inducing if not args.smoke_test else 100
    
    if args.n_inducing > 2_000:
    
        inducing = kmeans2(data, n_inducing, minit="points")[0]
    else:
        inducing = data[:args.n_inducing]
    
    return inducing
    
    
def get_kernel(args):
    
    # get kernel
    if args.kernel == "linear":
        kernel = gpytorch.kernels.Linear()
    elif args.kernel == "rbf":
        kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=None
        )
    elif args.kernel == "ard":
        kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=args.in_dim
        )
    elif args.kernel == "matern12":
        kernel = gpytorch.kernels.MaternKernel(
            nu=0.5,
            ard_num_dims=args.in_dim
        )
    elif args.kernel == "matern25":
        kernel = gpytorch.kernels.MaternKernel(
            nu=1.5,
            ard_num_dims=args.in_dim
        )
    elif args.kernel == "matern35":
        kernel = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=args.in_dim
        )
    elif args.kernel == "rff":
        kernel = gpytorch.kernels.RFFKernel(
            args.num_rff_samples
        )
    elif args.kernel == "mixture":
        kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=args.mixtures,
            ard_num_dims=args.in_dim,
        )
    elif args.kernel == "rq":
        kernel = gpytorch.kernels.RQKernel(
            ard_num_dims=args.in_dim
        )
    elif args.kernel == "delta":
        kernel = gpytorch.kernels.SpectralDeltaKernel(
            num_deltas=args.num_deltas,
        )
    else:
        raise ValueError(f"Unrecognized kernel: {args.kernel}")
    
    
    # add the kernel variance
    if args.kernel_variance:
        kernel = gpytorch.kernels.ScaleKernel(kernel)    
    
    
    return kernel


def get_likelihood(args):
    
    if args.likelihood == "gaussian":
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    else:
        raise ValueError(f"Unrecognized likelihood: {args.likelihood}")
        
    return likelihood


def get_variational_dist(inducing_points: torch.Tensor, args):
    
    if not args.variational_tril:
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(0)
        )
    else:
        variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(
            inducing_points.size(0)
        )
    
    return variational_distribution


# def get_variational_strategy(variational_dist, args):
    
#     if args.n_inducing > 5_000:
#         variationa