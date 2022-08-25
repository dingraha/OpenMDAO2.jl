using OpenMDAO2
using OpenMDAOCore: OpenMDAOCore
using PythonCall
using Test

@testset "Explicit Components" begin

    @testset "simple" begin
        struct ECompSimple <: OpenMDAOCore.AbstractExplicitComp end

        function OpenMDAOCore.setup(self::ECompSimple)
            input_data = [OpenMDAOCore.VarData("x")]
            output_data = [OpenMDAOCore.VarData("y")]
            partials_data = [OpenMDAOCore.PartialsData("y", "x")]

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompSimple, inputs, outputs)
            outputs["y"][1] = 2*inputs["x"][1]^2 + 1
            return nothing
        end

        function OpenMDAOCore.compute_partials!(self::ECompSimple, inputs, partials)
            partials["y", "x"][1] = 4*inputs["x"][1]
            return nothing
        end

        p = om.Problem()
        ecomp = ECompSimple()
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.run_model()

        # Check that the outputs are what we expect.
        # expected = 2*PyArray(p.get_val("x"))[1]^2 + 1
        # actual = PyArray(p.get_val("y"))[1]
        expected = 2 .* PyArray(p.get_val("x")).^2 .+ 1
        actual = PyArray(p.get_val("y"))
        @test actual ≈ expected

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are what we expect.
        ecomp_partials = pyconvert(Dict, cpd["ecomp"])
        actual = pyconvert(Dict, ecomp_partials["y", "x"])["J_fwd"]
        expected = 4 .* PyArray(p.get_val("x"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @testset "with option" begin
        struct ECompWithOption <: OpenMDAOCore.AbstractExplicitComp
            a::Float64
        end

        function OpenMDAOCore.setup(self::ECompWithOption)
            input_data = [OpenMDAOCore.VarData("x")]
            output_data = [OpenMDAOCore.VarData("y")]
            partials_data = [OpenMDAOCore.PartialsData("y", "x")]

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompWithOption, inputs, outputs)
            outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
            return nothing
        end

        function OpenMDAOCore.compute_partials!(self::ECompWithOption, inputs, partials)
            partials["y", "x"][1] = 4*self.a*inputs["x"][1]
            return nothing
        end

        p = om.Problem()
        a = 0.5
        ecomp = ECompWithOption(a)
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x"], promotes_outputs=["y"])
        p.setup(force_alloc_complex=true)
        p.set_val("x", 3.0)
        p.run_model()

        # Check that the outputs are what we expect.
        expected = 2 .* a.*PyArray(p.get_val("x")).^2 .+ 1
        actual = PyArray(p.get_val("y"))
        @test actual ≈ expected

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that the partials the user provided are what we expect.
        ecomp_partials = pyconvert(Dict, cpd["ecomp"])
        actual = pyconvert(Dict, ecomp_partials["y", "x"])["J_fwd"]
        expected = 4 .* a.*PyArray(p.get_val("x"))
        @test actual ≈ expected

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end

    @testset "matrix-free" begin
        struct ECompMatrixFree <: OpenMDAOCore.AbstractExplicitComp
            nrows::Int
            ncols::Int
        end

        function OpenMDAOCore.setup(self::ECompMatrixFree)
            nrows = self.nrows
            ncols = self.ncols
            input_data = [OpenMDAOCore.VarData("x1"; shape=(nrows, ncols)), OpenMDAOCore.VarData("x2"; shape=(self.nrows, self.ncols))]
            output_data = [OpenMDAOCore.VarData("y1"; shape=(nrows, ncols)), OpenMDAOCore.VarData("y2"; shape=(self.nrows, self.ncols))]
            partials_data = [OpenMDAOCore.PartialsData("*", "*")]  # I think this should work.

            return input_data, output_data, partials_data
        end

        function OpenMDAOCore.compute!(self::ECompMatrixFree, inputs, outputs)
            x1, x2 = inputs["x1"], inputs["x2"]
            y1, y2 = outputs["y1"], outputs["y2"]
            @. y1 = 2*x1 + 3*x2^2
            @. y2 = 4*x1^3 + 5*x2^4
            return nothing
        end

        function OpenMDAOCore.compute_jacvec_product!(self::ECompMatrixFree, inputs, d_inputs, d_outputs, mode)
            x1, x2 = inputs["x1"], inputs["x2"]
            x1dot = get(d_inputs, "x1", nothing)
            x2dot = get(d_inputs, "x2", nothing)
            y1dot = get(d_outputs, "y1", nothing)
            y2dot = get(d_outputs, "y2", nothing)
            if mode == "fwd"
                # For forward mode, we are tracking the derivatives of everything with
                # respect to upstream inputs, and our goal is to calculate the
                # derivatives of this components outputs wrt the upstream inputs given
                # the derivatives of inputs wrt the upstream inputs.
                if y1dot !== nothing
                    fill!(y1dot, 0)
                    if x1dot !== nothing
                        @. y1dot += 2*x1dot
                    end
                    if x2dot !== nothing
                        @. y1dot += 6*x2*x2dot
                    end
                end
                if y2dot !== nothing
                    fill!(y2dot, 0)
                    if x1dot !== nothing
                        @. y2dot += 12*x1^2*x1dot
                    end
                    if x2dot !== nothing
                        @. y2dot += 20*x2^3*x2dot
                    end
                end
            elseif mode == "rev"
                # For reverse mode, we are tracking the derivatives of everything with
                # respect to a downstream output, and our goal is to calculate the
                # derivatives of the downstream output wrt each input given the
                # derivatives of the downstream output wrt each output.
                #
                # So, let's say I have a function f(y1, y2).
                # I start with fdot = df/df = 1.
                # Then I say that y1dot = df/dy1 = fdot*df/dy1
                # and y2dot = df/dy2 = fdot*df/dy2
                # Hmm...
                # f(y1(x1,x2), y2(x1, x2)) = df/dy1*(dy1/dx1 + dy1/dx2) + df/dy2*(dy2/dx1 + dy2/dx2)
                if x1dot !== nothing
                    fill!(x1dot, 0)
                    if y1dot !== nothing
                        @. x1dot += y1dot*2
                    end
                    if x2dot !== nothing
                        @. x1dot += y2dot*(12*x1^2)
                    end
                end
                if x2dot !== nothing
                    fill!(x2dot, 0)
                    if y1dot !== nothing
                        @. x2dot += y1dot*(6*x2)
                    end
                    if y2dot !== nothing
                        @. x2dot += y2dot*(20*x2^3)
                    end
                end
            end
            return nothing
        end

        p = om.Problem()
        nrows, ncols = 2, 3
        ecomp = ECompMatrixFree(nrows, ncols)
        comp = make_component(ecomp)
        p.model.add_subsystem("ecomp", comp, promotes_inputs=["x1", "x2"], promotes_outputs=["y1", "y2"])
        p.setup(force_alloc_complex=true)
        p.set_val("x1", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 0.5)
        p.set_val("x2", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 1.0)
        p.run_model()

        # Test that the outputs are what we expect.
        expected = 2 .* PyArray(p.get_val("x1")) .+ 3 .* PyArray(p.get_val("x2")).^2
        actual = PyArray(p.get_val("y1"))
        @test expected ≈ actual

        expected = 4 .* PyArray(p.get_val("x1")).^3 .+ 5 .* PyArray(p.get_val("x2")).^4
        actual = PyArray(p.get_val("y2"))
        @test expected ≈ actual

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                @test PyArray(cpd_comp_var_wrt["J_rev"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

        p.set_val("x1", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 4.0)
        p.set_val("x2", reshape(0:(nrows*ncols-1), nrows, ncols) .+ 5.0)

        cpd = pyconvert(Dict, p.check_partials(compact_print=true, out_stream=nothing, method="cs"))

        # Check that partials approximated by the complex-step method match the user-provided partials.
        for comp in keys(cpd)
            for (pyvar, pywrt) in keys(cpd[comp])
                var = pyconvert(Any, pyvar)
                wrt = pyconvert(Any, pywrt)
                cpd_comp = pyconvert(PyDict{Tuple{String, String}}, cpd[comp])
                cpd_comp_var_wrt = pyconvert(PyDict{String}, cpd_comp[var, wrt])
                @test PyArray(cpd_comp_var_wrt["J_fwd"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
                @test PyArray(cpd_comp_var_wrt["J_rev"]) ≈ PyArray(cpd_comp_var_wrt["J_fd"])
            end
        end

    end
end
