import gpflow
from scipy.cluster.vq import kmeans2

def add_model_args(parser):
    parser.add_argument('--model', type=str, default="svgp")
    parser.add_argument("--n-inducing", type=int, default=100)
    
    # likelihood params
    parser.add_argument('--likelihood', type=str, default="gaussian")
    parser.add_argument('--noise-scale', type=float, default=0.05)
    
    # kernel params
    parser.add_argument('--kernel', type=str, default="rbf")
    parser.add_argument('--lon-scale', type=float, default=1.0)
    parser.add_argument('--lat-scale', type=float, default=1.0)
    parser.add_argument('--time-scale', type=float, default=7.0)
    parser.add_argument("--variance", type=float, default=1.0)

    return parser


def get_likelihood(args):
    if args.likelihood == "gaussian":
        likelihood = gpflow.likelihoods.Gaussian(variance=args.noise_scale)
    else:
        raise ValueError(f"Unrecognized likelihood: {args.likelihood}")
    return likelihood

def get_kernel(args):
    
    if args.kernel == "rbf":
        kernel = gpflow.kernels.SquaredExponential(
            lengthscales=[args.time_scale, args.lon_scale, args.lat_scale],
            variance=args.variance
        )
    else:
        raise ValueError(f"Unrecognized kernel entry: {args.kernel}")
        
    return kernel
        
    
def get_inducing_points(data, args):
    
    inducing = kmeans2(data, args.n_inducing, minit="points")[0]
    
    return inducing
    
    