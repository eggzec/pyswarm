C     ========================================================
C            Particle Swarm Optimisation Algorithm
C
C     A gradient-free, evolutionary optimization
C     algorithm.
C     ========================================================
C
C     Steps performed:
C       1. Evaluate initial particle positions via EVAL callback
C       2. Bootstrap personal-best and global-best arrays
C       3. Main loop  IT = 1 .. MAXITER:
C            a. Velocity update  (PSO inertia formula)
C            b. Position update + bound clamping
C            c. Integer variable rounding  (if NIVAR > 0)
C            d. Evaluate objectives via EVAL callback
C            e. Update personal bests
C            f. Update global best + convergence checks
C            g. Patience (stagnation) check
C
C     EVAL Callback Contract
C     ----------------------
C     EVAL is declared EXTERNAL and called as:
C
C         FX_I = EVAL(XTMP, NDIM)
C
C     XTMP(NDIM) is a contiguous copy of particle I's position.
C     Return value convention:
C        < 1.0D300  =>  feasible; return value IS the objective
C       >= 1.0D300  =>  infeasible (treated as +infinity)
C
C     All feasibility logic is encoded in the return value;
C     no separate constraint callback is required.
C
C     Random numbers
C     --------------
C     Generated in Python for reproducible seeding, then passed as:
C         RND_P(MAXITER, NSWARM, NDIM)  cognitive random factors
C         RND_G(MAXITER, NSWARM, NDIM)  social    random factors
C
C     Initial positions X and velocities V are also computed in Python
C     and passed as in/out dummy arrays (valid F77 adjustable arrays).
C
C     Argument list
C     -------------
C     EVAL    EXTERNAL DOUBLE PRECISION FUNCTION  Python callback
C     LB      (NDIM)              in    lower bounds
C     UB      (NDIM)              in    upper bounds
C     X       (NSWARM, NDIM)      in/out initial positions -> final
C     V       (NSWARM, NDIM)      in/out initial velocities -> final
C     RND_P   (MAXITER,NSWARM,NDIM) in  cognitive random factors [0,1]
C     RND_G   (MAXITER,NSWARM,NDIM) in  social    random factors [0,1]
C     OMEGA               in    inertia weight
C     PHIP                in    cognitive coefficient
C     PHIG                in    social coefficient
C     MAXITER   [hidden]  in    maximum iterations
C     MINSTEP             in    step-size convergence threshold
C     MINFUNC             in    function-change convergence threshold
C     PATIENCE            in    stagnation limit (0 = disabled)
C     DEBUG               in    print to stdout (1=on, 0=off)
C     NSWARM    [hidden]  in    swarm size  (from X shape)
C     NDIM      [hidden]  in    design-var count (from LB shape)
C     IVAR      (NIVAR)   in    1-based integer-variable indices
C     NIVAR     [hidden]  in    number of integer variables
C     XTMP      (NDIM)  [hidden workspace]  contiguous row buffer
C     FX        (NSWARM)[hidden workspace]  current objective values
C     G         (NDIM)   out   global-best position
C     FG                 out   global-best value (>=1D300 if no sol.)
C     P         (NSWARM,NDIM) out  per-particle best positions
C     FP        (NSWARM) out   per-particle best objective values
C     NITERS             out   iterations performed
C     STATUS             out   0=maxiter 1=minstep 2=minfunc 3=patience
C     =================================================================
      SUBROUTINE PSO(EVAL,
     &               LB, UB, X, V, RND_P, RND_G,
     &               OMEGA, PHIP, PHIG,
     &               MAXITER, MINSTEP, MINFUNC, PATIENCE, DEBUG,
     &               NSWARM, NDIM,
     &               IVAR, NIVAR,
     &               XTMP, FX,
     &               G, FG, P, FP,
     &               NITERS, STATUS)
      IMPLICIT NONE

C     --- External callback ---
      DOUBLE PRECISION EVAL
      EXTERNAL         EVAL

C     --- Dimension scalars (all are dummy args so dims are adjustable) ---
      INTEGER NSWARM, NDIM, NIVAR, MAXITER, PATIENCE, DEBUG

C     --- Bound arrays ---
      DOUBLE PRECISION LB(NDIM), UB(NDIM)

C     --- Particle state (in/out) ---
      DOUBLE PRECISION X(NSWARM, NDIM), V(NSWARM, NDIM)

C     --- Pre-generated random arrays for every iteration ---
      DOUBLE PRECISION RND_P(MAXITER, NSWARM, NDIM)
      DOUBLE PRECISION RND_G(MAXITER, NSWARM, NDIM)

C     --- Integer variable indices (1-based) ---
      INTEGER IVAR(NIVAR)

C     --- PSO parameters ---
      DOUBLE PRECISION OMEGA, PHIP, PHIG, MINSTEP, MINFUNC

C     --- Workspace arrays (hidden from Python, allocated by f2py) ---
      DOUBLE PRECISION XTMP(NDIM), FX(NSWARM)

C     --- Output arrays ---
      DOUBLE PRECISION G(NDIM), FG
      DOUBLE PRECISION P(NSWARM, NDIM), FP(NSWARM)
      INTEGER          NITERS, STATUS

C     --- Local scalars ---
      INTEGER          I, IT, J, K, JJ, IMIN, STAGNATION, IMPROVED
      DOUBLE PRECISION FPMIN, STEPSIZE, FUNCDIFF, DIFF, VAL

C     --- Named constants for infeasibility sentinel ---
      DOUBLE PRECISION INFEAS, LARGE
      PARAMETER        (INFEAS = 1.0D300, LARGE = 2.0D300)

C     =================================================================
C     STEP 1 -- Evaluate initial particle positions
C     =================================================================
      DO I = 1, NSWARM
          DO J = 1, NDIM
              XTMP(J) = X(I,J)
          END DO
          FX(I) = EVAL(XTMP, NDIM)
      END DO

C     =================================================================
C     STEP 2 -- Bootstrap personal bests
C     (FP initialised to LARGE; only feasible particles update it)
C     =================================================================
      DO I = 1, NSWARM
          FP(I) = LARGE
          DO J = 1, NDIM
              P(I,J) = X(I,J)
          END DO
      END DO
      DO I = 1, NSWARM
          IF (FX(I) .LT. INFEAS .AND. FX(I) .LT. FP(I)) THEN
              FP(I) = FX(I)
              DO J = 1, NDIM
                  P(I,J) = X(I,J)
              END DO
          END IF
      END DO

C     =================================================================
C     STEP 3 -- Bootstrap global best
C     =================================================================
      FG    = LARGE
      FPMIN = FP(1)
      IMIN  = 1
      DO I = 2, NSWARM
          IF (FP(I) .LT. FPMIN) THEN
              FPMIN = FP(I)
              IMIN  = I
          END IF
      END DO
      IF (FPMIN .LT. INFEAS) THEN
          FG = FPMIN
          DO J = 1, NDIM
              G(J) = P(IMIN,J)
          END DO
      ELSE
C         No feasible starting point -- use particle 1 as fallback
          DO J = 1, NDIM
              G(J) = X(1,J)
          END DO
      END IF

C     =================================================================
C     MAIN ITERATION LOOP
C     =================================================================
      IT         = 1
      STAGNATION = 0
      STATUS     = 0

100   CONTINUE
      IF (IT .GT. MAXITER) GOTO 900

C     -----------------------------------------------------------------
C     (a) Velocity update: V = omega*V + phip*rp*(P-X) + phig*rg*(G-X)
C     -----------------------------------------------------------------
      DO I = 1, NSWARM
          DO J = 1, NDIM
              V(I,J) = OMEGA * V(I,J)
     &               + PHIP * RND_P(IT,I,J) * (P(I,J) - X(I,J))
     &               + PHIG * RND_G(IT,I,J) * (G(J)   - X(I,J))
          END DO
      END DO

C     -----------------------------------------------------------------
C     (b) Position update + bound clamping: X = clip(X + V, LB, UB)
C     -----------------------------------------------------------------
      DO I = 1, NSWARM
          DO J = 1, NDIM
              X(I,J) = X(I,J) + V(I,J)
              IF (X(I,J) .LT. LB(J)) X(I,J) = LB(J)
              IF (X(I,J) .GT. UB(J)) X(I,J) = UB(J)
          END DO
      END DO

C     -----------------------------------------------------------------
C     (c) Integer variable rounding (skip if NIVAR == 0)
C     -----------------------------------------------------------------
      IF (NIVAR .GT. 0) THEN
          DO I = 1, NSWARM
              DO K = 1, NIVAR
                  JJ  = IVAR(K)
                  VAL = DNINT(X(I,JJ))
                  IF (VAL .LT. LB(JJ)) VAL = LB(JJ)
                  IF (VAL .GT. UB(JJ)) VAL = UB(JJ)
                  X(I,JJ) = VAL
              END DO
          END DO
      END IF

C     -----------------------------------------------------------------
C     (d) Evaluate objectives via Python callback
C     -----------------------------------------------------------------
      DO I = 1, NSWARM
          DO J = 1, NDIM
              XTMP(J) = X(I,J)
          END DO
          FX(I) = EVAL(XTMP, NDIM)
      END DO

C     -----------------------------------------------------------------
C     (e) Update personal bests (feasible particles only)
C     -----------------------------------------------------------------
      DO I = 1, NSWARM
          IF (FX(I) .LT. INFEAS .AND. FX(I) .LT. FP(I)) THEN
              FP(I) = FX(I)
              DO J = 1, NDIM
                  P(I,J) = X(I,J)
              END DO
          END IF
      END DO

C     -----------------------------------------------------------------
C     Find the best personal-best across the swarm
C     -----------------------------------------------------------------
      FPMIN = FP(1)
      IMIN  = 1
      DO I = 2, NSWARM
          IF (FP(I) .LT. FPMIN) THEN
              FPMIN = FP(I)
              IMIN  = I
          END IF
      END DO

C     -----------------------------------------------------------------
C     (f) Update global best + convergence checks
C     -----------------------------------------------------------------
      IMPROVED = 0
      IF (FPMIN .LT. FG .AND. FPMIN .LT. INFEAS) THEN
          IMPROVED   = 1
          FUNCDIFF   = DABS(FG - FPMIN)
          STEPSIZE   = 0.0D0
          DO J = 1, NDIM
              DIFF     = G(J) - P(IMIN,J)
              STEPSIZE = STEPSIZE + DIFF * DIFF
          END DO
          STEPSIZE   = DSQRT(STEPSIZE)
          DO J = 1, NDIM
              G(J) = P(IMIN,J)
          END DO
          FG         = FPMIN
          STAGNATION = 0

          IF (DEBUG .NE. 0) WRITE(*,*) 'New best iter',IT,':',FG

C         minfunc check
          IF (FUNCDIFF .LE. MINFUNC) THEN
              STATUS = 2
              IF (DEBUG .NE. 0)
     &            WRITE(*,*) 'Stop: func improvement <= MINFUNC'
              GOTO 900
          END IF

C         minstep check
          IF (STEPSIZE .LE. MINSTEP) THEN
              STATUS = 1
              IF (DEBUG .NE. 0)
     &            WRITE(*,*) 'Stop: step size <= MINSTEP'
              GOTO 900
          END IF

      ELSE
C         -----------------------------------------------------------------
C         (g) Patience (stagnation) check
C         -----------------------------------------------------------------
          STAGNATION = STAGNATION + 1
          IF (PATIENCE .GT. 0 .AND. STAGNATION .GE. PATIENCE) THEN
              STATUS = 3
              IF (DEBUG .NE. 0)
     &            WRITE(*,*) 'Stop: patience exhausted'
              GOTO 900
          END IF
      END IF

      IF (DEBUG .NE. 0) WRITE(*,*) 'Iter', IT, 'best =', FG

      IT = IT + 1
      GOTO 100

C     =================================================================
900   CONTINUE
      NITERS = IT - 1
      RETURN
      END
