
// Code generated by stanc v2.27.0
#include <stan/model/model_header.hpp>
namespace deseq2_single_model_namespace {

using stan::io::dump;
using stan::model::assign;
using stan::model::index_uni;
using stan::model::index_max;
using stan::model::index_min;
using stan::model::index_min_max;
using stan::model::index_multi;
using stan::model::index_omni;
using stan::model::model_base_crtp;
using stan::model::rvalue;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 32> locations_array__ = 
{" (found before start of program)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 13, column 2 to column 15)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 14, column 2 to column 12)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 15, column 2 to column 27)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 32, column 2 to column 22)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 33, column 2 to column 22)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 35, column 4 to column 12)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 36, column 4 to column 62)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 37, column 4 to column 64)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 38, column 4 to column 72)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 34, column 16 to line 39, column 3)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 34, column 2 to line 39, column 3)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 20, column 2 to column 43)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 21, column 2 to column 47)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 22, column 2 to column 22)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 25, column 4 to column 12)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 26, column 4 to column 62)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 27, column 4 to column 52)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 24, column 16 to line 28, column 3)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 24, column 2 to line 28, column 3)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 2, column 2 to column 17)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 3, column 2 to column 17)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 4, column 12 to column 13)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 4, column 2 to column 15)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 5, column 8 to column 9)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 5, column 2 to column 11)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 6, column 8 to column 9)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 6, column 2 to column 11)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 7, column 2 to column 19)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 8, column 2 to column 21)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 32, column 9 to column 10)",
 " (in '/mnt/ceph/users/djin/git/q2-differential/q2_differential/assets/deseq2_single.stan', line 33, column 9 to column 10)"};



class deseq2_single_model final : public model_base_crtp<deseq2_single_model> {

 private:
  int N;
  int D;
  std::vector<double> slog;
  std::vector<int> M;
  std::vector<int> y;
  double control_loc;
  double control_scale; 
  
 
 public:
  ~deseq2_single_model() { }
  
  inline std::string model_name() const final { return "deseq2_single_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.27.0", "stancflags = "};
  }
  
  
  deseq2_single_model(stan::io::var_context& context__,
                      unsigned int random_seed__ = 0,
                      std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "deseq2_single_model_namespace::deseq2_single_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      current_statement__ = 20;
      context__.validate_dims("data initialization","N","int",
           std::vector<size_t>{});
      N = std::numeric_limits<int>::min();
      
      current_statement__ = 20;
      N = context__.vals_i("N")[(1 - 1)];
      current_statement__ = 20;
      check_greater_or_equal(function__, "N", N, 0);
      current_statement__ = 21;
      context__.validate_dims("data initialization","D","int",
           std::vector<size_t>{});
      D = std::numeric_limits<int>::min();
      
      current_statement__ = 21;
      D = context__.vals_i("D")[(1 - 1)];
      current_statement__ = 21;
      check_greater_or_equal(function__, "D", D, 0);
      current_statement__ = 22;
      validate_non_negative_index("slog", "N", N);
      current_statement__ = 23;
      context__.validate_dims("data initialization","slog","double",
           std::vector<size_t>{static_cast<size_t>(N)});
      slog = std::vector<double>(N, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 23;
      slog = context__.vals_r("slog");
      current_statement__ = 24;
      validate_non_negative_index("M", "N", N);
      current_statement__ = 25;
      context__.validate_dims("data initialization","M","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      M = std::vector<int>(N, std::numeric_limits<int>::min());
      
      current_statement__ = 25;
      M = context__.vals_i("M");
      current_statement__ = 26;
      validate_non_negative_index("y", "N", N);
      current_statement__ = 27;
      context__.validate_dims("data initialization","y","int",
           std::vector<size_t>{static_cast<size_t>(N)});
      y = std::vector<int>(N, std::numeric_limits<int>::min());
      
      current_statement__ = 27;
      y = context__.vals_i("y");
      current_statement__ = 28;
      context__.validate_dims("data initialization","control_loc","double",
           std::vector<size_t>{});
      control_loc = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 28;
      control_loc = context__.vals_r("control_loc")[(1 - 1)];
      current_statement__ = 29;
      context__.validate_dims("data initialization","control_scale","double",
           std::vector<size_t>{});
      control_scale = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 29;
      control_scale = context__.vals_r("control_scale")[(1 - 1)];
      current_statement__ = 30;
      validate_non_negative_index("y_predict", "N", N);
      current_statement__ = 31;
      validate_non_negative_index("log_lhood", "N", N);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    num_params_r__ = 1 + 1 + 2;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "deseq2_single_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      local_scalar_t__ control;
      control = DUMMY_VAR__;
      
      current_statement__ = 1;
      control = in__.template read<local_scalar_t__>();
      local_scalar_t__ beta;
      beta = DUMMY_VAR__;
      
      current_statement__ = 2;
      beta = in__.template read<local_scalar_t__>();
      Eigen::Matrix<local_scalar_t__, -1, 1> alpha;
      alpha = Eigen::Matrix<local_scalar_t__, -1, 1>(2);
      stan::math::fill(alpha, DUMMY_VAR__);
      
      current_statement__ = 3;
      alpha = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                0, lp__, 2);
      {
        current_statement__ = 12;
        lp_accum__.add(
          lognormal_lpdf<propto__>(to_vector(alpha), stan::math::log(10), 1));
        current_statement__ = 13;
        lp_accum__.add(
          normal_lpdf<propto__>(control, control_loc, control_scale));
        current_statement__ = 14;
        lp_accum__.add(normal_lpdf<propto__>(beta, 0, 5));
        current_statement__ = 19;
        for (int i = 1; i <= N; ++i) {
          local_scalar_t__ mu;
          mu = DUMMY_VAR__;
          
          current_statement__ = 16;
          mu = (rvalue(slog, "slog", index_uni(i)) +
                 log_inv_logit(
                   (control + ((rvalue(M, "M", index_uni(i)) - 1) * beta))));
          current_statement__ = 17;
          lp_accum__.add(
            neg_binomial_2_log_lpmf<propto__>(rvalue(y, "y", index_uni(i)),
              mu,
              inv(
                rvalue(alpha, "alpha",
                  index_uni(rvalue(M, "M", index_uni(i)))))));
        }
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_std_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.resize(0);
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "deseq2_single_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      double control;
      control = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      control = in__.template read<local_scalar_t__>();
      double beta;
      beta = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 2;
      beta = in__.template read<local_scalar_t__>();
      Eigen::Matrix<double, -1, 1> alpha;
      alpha = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(alpha, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 3;
      alpha = in__.template read_constrain_lb<Eigen::Matrix<local_scalar_t__, -1, 1>, jacobian__>(
                0, lp__, 2);
      vars__.emplace_back(control);
      vars__.emplace_back(beta);
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(alpha[(sym1__ - 1)]);
      }
      if (logical_negation((primitive_value(emit_transformed_parameters__) ||
            primitive_value(emit_generated_quantities__)))) {
        return ;
      } 
      if (logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      Eigen::Matrix<double, -1, 1> y_predict;
      y_predict = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(y_predict, std::numeric_limits<double>::quiet_NaN());
      
      Eigen::Matrix<double, -1, 1> log_lhood;
      log_lhood = Eigen::Matrix<double, -1, 1>(N);
      stan::math::fill(log_lhood, std::numeric_limits<double>::quiet_NaN());
      
      current_statement__ = 11;
      for (int n = 1; n <= N; ++n) {
        double mu;
        mu = std::numeric_limits<double>::quiet_NaN();
        
        current_statement__ = 7;
        mu = (rvalue(slog, "slog", index_uni(n)) +
               log_inv_logit(
                 (control + ((rvalue(M, "M", index_uni(n)) - 1) * beta))));
        current_statement__ = 8;
        assign(y_predict,
          neg_binomial_2_log_rng(mu,
            inv(
              rvalue(alpha, "alpha", index_uni(rvalue(M, "M", index_uni(n))))),
            base_rng__), "assigning variable y_predict", index_uni(n));
        current_statement__ = 9;
        assign(log_lhood,
          neg_binomial_2_log_lpmf<false>(rvalue(y, "y", index_uni(n)), mu,
            inv(
              rvalue(alpha, "alpha", index_uni(rvalue(M, "M", index_uni(n)))))),
          "assigning variable log_lhood", index_uni(n));
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(y_predict[(sym1__ - 1)]);
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        vars__.emplace_back(log_lhood[(sym1__ - 1)]);
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_std_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(const stan::io::var_context& context__,
                                   VecI& params_i__, VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    vars__.clear();
    vars__.reserve(num_params_r__);
    int current_statement__ = 0; 
    
    try {
      int pos__;
      pos__ = std::numeric_limits<int>::min();
      
      pos__ = 1;
      double control;
      control = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 1;
      control = context__.vals_r("control")[(1 - 1)];
      double beta;
      beta = std::numeric_limits<double>::quiet_NaN();
      
      current_statement__ = 2;
      beta = context__.vals_r("beta")[(1 - 1)];
      Eigen::Matrix<double, -1, 1> alpha;
      alpha = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(alpha, std::numeric_limits<double>::quiet_NaN());
      
      {
        std::vector<local_scalar_t__> alpha_flat__;
        current_statement__ = 3;
        alpha_flat__ = context__.vals_r("alpha");
        current_statement__ = 3;
        pos__ = 1;
        current_statement__ = 3;
        for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
          current_statement__ = 3;
          assign(alpha, alpha_flat__[(pos__ - 1)],
            "assigning variable alpha", index_uni(sym1__));
          current_statement__ = 3;
          pos__ = (pos__ + 1);
        }
      }
      Eigen::Matrix<double, -1, 1> alpha_free__;
      alpha_free__ = Eigen::Matrix<double, -1, 1>(2);
      stan::math::fill(alpha_free__, std::numeric_limits<double>::quiet_NaN());
      
      
      current_statement__ = 3;
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        current_statement__ = 3;
        assign(alpha_free__, stan::math::lb_free(alpha[(sym1__ - 1)], 0),
          "assigning variable alpha_free__", index_uni(sym1__));
      }
      vars__.emplace_back(control);
      vars__.emplace_back(beta);
      for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
        vars__.emplace_back(alpha_free__[(sym1__ - 1)]);
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
      // Next line prevents compiler griping about no return
      throw std::runtime_error("*** IF YOU SEE THIS, PLEASE REPORT A BUG ***"); 
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"control", "beta", "alpha",
      "y_predict", "log_lhood"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{},
      std::vector<size_t>{}, std::vector<size_t>{static_cast<size_t>(2)},
      std::vector<size_t>{static_cast<size_t>(N)},
      std::vector<size_t>{static_cast<size_t>(N)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "control");
    param_names__.emplace_back(std::string() + "beta");
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "alpha" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_predict" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "log_lhood" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    param_names__.emplace_back(std::string() + "control");
    param_names__.emplace_back(std::string() + "beta");
    for (int sym1__ = 1; sym1__ <= 2; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "alpha" + '.' + std::to_string(sym1__));
      } 
    }
    if (emit_transformed_parameters__) {
      
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "y_predict" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= N; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "log_lhood" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"control\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"alpha\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"y_predict\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lhood\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N) + "},\"block\":\"generated_quantities\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"control\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"beta\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"alpha\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(2) + "},\"block\":\"parameters\"},{\"name\":\"y_predict\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N) + "},\"block\":\"generated_quantities\"},{\"name\":\"log_lhood\",\"type\":{\"name\":\"vector\",\"length\":" + std::to_string(N) + "},\"block\":\"generated_quantities\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      std::vector<double> vars_vec;
      vars_vec.reserve(vars.size());
      std::vector<int> params_i;
      write_array_impl(base_rng, params_r, params_i, vars_vec,
          emit_transformed_parameters, emit_generated_quantities, pstream);
      vars = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        vars_vec.data(), vars_vec.size());
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      write_array_impl(base_rng, params_r, params_i, vars,
       emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec;
      params_r_vec.reserve(params_r.size());
      std::vector<int> params_i;
      transform_inits_impl(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }
    inline void transform_inits(const stan::io::var_context& context,
                                std::vector<int>& params_i,
                                std::vector<double>& vars,
                                std::ostream* pstream = nullptr) const final {
      transform_inits_impl(context, params_i, vars, pstream);
    }

};
}
using stan_model = deseq2_single_model_namespace::deseq2_single_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return deseq2_single_model_namespace::profiles__;
}

#endif


