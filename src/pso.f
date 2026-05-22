C     =================================================================
C     FILE: pso.f
C     Complete Particle Swarm Optimisation -- Fortran 77
C
C     SUBROUTINE PSO
C     ==============
C     Performs ONE complete PSO iteration step:
C
C       (1) Update personal-best positions and objective values
C       (2) Update global-best position and objective value;
C             compute step-size and function-improvement metrics
C       (3) Update particle velocities (inertia formula)
C       (4) Advance particle positions and clamp to [lb, ub]
C       (5) Round integer design variables (if NIVAR > 0)
C
C     The Python host drives the outer loop, evaluates the objective
C     function and constraints, supplies pre-generated uniform random
C     arrays RP and RG, and inspects STEPSIZE / FUNCDIFF / IMPROVED
C     to decide when to terminate.
C
C     Arguments
C     ---------
C     X       DOUBLE (NSWARM x NDIM)  in/out  particle positions
C     V       DOUBLE (NSWARM x NDIM)  in/out  particle velocities
C     P       DOUBLE (NSWARM x NDIM)  in/out  personal-best positions
C     FP      DOUBLE (NSWARM)         in/out  personal-best obj values
C     G       DOUBLE (NDIM)           in/out  global-best position
C     FG      DOUBLE                  in/out  global-best obj value
C     FX      DOUBLE (NSWARM)         in      current obj values
C     FS      INTEGER(NSWARM)         in      feasibility (1=OK, 0=not)
C     RP      DOUBLE (NSWARM x NDIM)  in      uniform[0,1] random (cogn.)
C     RG      DOUBLE (NSWARM x NDIM)  in      uniform[0,1] random (soc.)
C     LB      DOUBLE (NDIM)           in      lower bounds
C     UB      DOUBLE (NDIM)           in      upper bounds
C     IVAR    INTEGER(NIVAR)          in      1-based int-var indices
C     OMEGA   DOUBLE                  in      inertia weight
C     PHIP    DOUBLE                  in      cognitive coefficient
C     PHIG    DOUBLE                  in      social coefficient
C     MINSTEP DOUBLE                  in      step-size convergence tol.
C     MINFUNC DOUBLE                  in      func-improvement tol.
C     NSWARM  INTEGER                 in      swarm size
C     NDIM    INTEGER                 in      number of design variables
C     NIVAR   INTEGER                 in      number of integer variables
C     STEPSIZE DOUBLE                 out     Euclidean move of G
C     FUNCDIFF DOUBLE                 out     |FG_old - FG_new|
C     IMPROVED INTEGER                out     1 if FG improved, else 0
C     =================================================================
      SUBROUTINE PSO(X, V, P, FP, G, FG, FX, FS,
     &               RP, RG, LB, UB, IVAR,
     &               OMEGA, PHIP, PHIG, MINSTEP, MINFUNC,
     &               NSWARM, NDIM, NIVAR,
     &               STEPSIZE, FUNCDIFF, IMPROVED)
      IMPLICIT NONE

C     --- Dimension arguments ---
      INTEGER NSWARM, NDIM, NIVAR

C     --- In / out arrays ---
      DOUBLE PRECISION X(NSWARM, NDIM), V(NSWARM, NDIM)
      DOUBLE PRECISION P(NSWARM, NDIM), FP(NSWARM)
      DOUBLE PRECISION G(NDIM), FG

C     --- Input arrays (current evaluation) ---
      DOUBLE PRECISION FX(NSWARM)
      INTEGER          FS(NSWARM)

C     --- Input arrays (random factors for this step) ---
      DOUBLE PRECISION RP(NSWARM, NDIM), RG(NSWARM, NDIM)

C     --- Input arrays (bounds & integer variable indices) ---
      DOUBLE PRECISION LB(NDIM), UB(NDIM)
      INTEGER          IVAR(NIVAR)

C     --- Scalar PSO parameters ---
      DOUBLE PRECISION OMEGA, PHIP, PHIG, MINSTEP, MINFUNC

C     --- Output convergence metrics ---
      DOUBLE PRECISION STEPSIZE, FUNCDIFF
      INTEGER          IMPROVED

C     --- Local variables ---
      INTEGER          I, J, K, IMIN
      DOUBLE PRECISION FPMIN, DIFF, VAL

C     =================================================================
C     STEP 1 -- Update personal bests
C     For each particle: if feasible and current value < personal best,
C     record the current position and value as the new personal best.
C     =================================================================
      DO I = 1, NSWARM
          IF (FS(I) .NE. 0 .AND. FX(I) .LT. FP(I)) THEN
              DO J = 1, NDIM
                  P(I,J) = X(I,J)
              END DO
              FP(I) = FX(I)
          END IF
      END DO

C     =================================================================
C     STEP 2 -- Update global best
C     Find the particle with the best personal-best value.
C     If it improves the global best, update G and FG and record
C     the convergence metrics.
C     =================================================================
      FPMIN = FP(1)
      IMIN  = 1
      DO I = 2, NSWARM
          IF (FP(I) .LT. FPMIN) THEN
              FPMIN = FP(I)
              IMIN  = I
          END IF
      END DO

      IMPROVED = 0
      STEPSIZE = 0.0D0
      FUNCDIFF = 0.0D0

      IF (FPMIN .LT. FG) THEN
          IMPROVED = 1
          FUNCDIFF = DABS(FG - FPMIN)
          DO J = 1, NDIM
              DIFF     = G(J) - P(IMIN,J)
              STEPSIZE = STEPSIZE + DIFF * DIFF
          END DO
          STEPSIZE = DSQRT(STEPSIZE)
          DO J = 1, NDIM
              G(J) = P(IMIN,J)
          END DO
          FG = FPMIN
      END IF

C     =================================================================
C     STEP 3 -- Velocity update (PSO inertia formula)
C
C       V = omega*V + phip*rp*(P - X) + phig*rg*(G - X)
C     =================================================================
      DO I = 1, NSWARM
          DO J = 1, NDIM
              V(I,J) = OMEGA * V(I,J)
     &               + PHIP * RP(I,J) * (P(I,J) - X(I,J))
     &               + PHIG * RG(I,J) * (G(J)   - X(I,J))
          END DO
      END DO

C     =================================================================
C     STEP 4 -- Position update and bound clamping
C
C       X = clip(X + V, lb, ub)
C     =================================================================
      DO I = 1, NSWARM
          DO J = 1, NDIM
              X(I,J) = X(I,J) + V(I,J)
              IF (X(I,J) .LT. LB(J)) X(I,J) = LB(J)
              IF (X(I,J) .GT. UB(J)) X(I,J) = UB(J)
          END DO
      END DO

C     =================================================================
C     STEP 5 -- Round integer design variables (only if NIVAR > 0)
C
C       For each integer variable j: X(:,j) = clip(round(X(:,j)), lb, ub)
C       IVAR holds 1-based Fortran column indices.
C     =================================================================
      IF (NIVAR .GT. 0) THEN
          DO I = 1, NSWARM
              DO K = 1, NIVAR
                  J   = IVAR(K)
                  VAL = DNINT(X(I,J))
                  IF (VAL .LT. LB(J)) VAL = LB(J)
                  IF (VAL .GT. UB(J)) VAL = UB(J)
                  X(I,J) = VAL
              END DO
          END DO
      END IF

      RETURN
      END
