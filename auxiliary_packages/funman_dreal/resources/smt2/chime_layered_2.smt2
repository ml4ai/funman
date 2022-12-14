(set-logic QF_LRA)
(declare-fun i_0 () Real)
(declare-fun s_0 () Real)
(declare-fun beta_0 () Real)
(declare-fun beta_1 () Real)
(declare-fun gamma () Real)
(declare-fun delta () Real)
(declare-fun n () Real)
(declare-fun r_0 () Real)
(assert (and (and (= s_0 1000.0) (= i_0 1.0) (= r_0 1.0) (= n (+ (+ s_0 i_0) r_0))) (and (= gamma 0.07142857142857142) (= delta 0.0) (= beta_0 6.7e-05) (= beta_1 6.7e-05))))
(push 1)
(assert (< i_0 (* n 0.01)))
(push 1)
(declare-fun s_n_1 () Real)
(declare-fun r_n_1 () Real)
(declare-fun i_n_1 () Real)
(assert (and (= r_n_1 (+ (* gamma i_0) r_0)) (= s_n_1 (+ (* (* (* beta_0 (- 1.0)) s_0) i_0) s_0)) (= i_n_1 (+ (- (* (* beta_0 s_0) i_0) (* gamma i_0)) i_0)) (<= r_n_1 n) (<= 0.0 r_n_1) (<= s_n_1 n) (<= 0.0 s_n_1) (<= i_n_1 n) (<= 0.0 i_n_1)))
(push 1)
(declare-fun scale_1 () Real)
(assert (and (= scale_1 (/ n (+ (+ s_n_1 i_n_1) r_n_1))) (<= scale_1 1.0) (<= 0.0 scale_1)))
(push 1)
(declare-fun i_1 () Real)
(declare-fun s_1 () Real)
(declare-fun r_1 () Real)
(assert (and (= s_1 (* s_n_1 scale_1)) (= i_1 (* i_n_1 scale_1)) (= r_1 (* r_n_1 scale_1)) (<= r_1 n) (<= 0.0 r_1) (<= s_1 n) (<= 0.0 s_1) (<= i_1 n) (<= 0.0 i_1)))
(push 1)
(assert (< i_1 (* n 0.01)))
(push 1)
(declare-fun r_n_2 () Real)
(declare-fun s_n_2 () Real)
(declare-fun i_n_2 () Real)
(assert (and (= r_n_2 (+ (* gamma i_1) r_1)) (= s_n_2 (+ (* (* (* beta_0 (- 1.0)) s_1) i_1) s_1)) (= i_n_2 (+ (- (* (* beta_0 s_1) i_1) (* gamma i_1)) i_1)) (<= r_n_2 n) (<= 0.0 r_n_2) (<= s_n_2 n) (<= 0.0 s_n_2) (<= i_n_2 n) (<= 0.0 i_n_2)))
(push 1)
(declare-fun scale_2 () Real)
(assert (and (= scale_2 (/ n (+ (+ s_n_2 i_n_2) r_n_2))) (<= scale_2 1.0) (<= 0.0 scale_2)))
(push 1)
(declare-fun i_2 () Real)
(declare-fun s_2 () Real)
(declare-fun r_2 () Real)
(assert (and (= s_2 (* s_n_2 scale_2)) (= i_2 (* i_n_2 scale_2)) (= r_2 (* r_n_2 scale_2)) (<= r_2 n) (<= 0.0 r_2) (<= s_2 n) (<= 0.0 s_2) (<= i_2 n) (<= 0.0 i_2)))
(push 1)
(assert (< i_2 (* n 0.01)))
(push 1)
(check-sat)
