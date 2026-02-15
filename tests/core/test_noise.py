"""Unit tests for noise schedule implementations in ``xlm.noise``.

Add tests for real noise schedules here as they are implemented / imported.
For example::

    class TestLogLinearNoiseSchedule:
        @pytest.fixture()
        def schedule(self):
            from idlm.noise_schedule import LogLinearNoiseSchedule
            return LogLinearNoiseSchedule(...)

        def test_total_noise_is_monotonic(self, schedule):
            t = torch.linspace(0, 1, 100)
            total = schedule.total_noise(t)
            assert (total[1:] >= total[:-1]).all()

        def test_noise_rate_non_negative(self, schedule):
            t = torch.linspace(0, 1, 100)
            rate = schedule.noise_rate(t)
            assert (rate >= 0).all()
"""
