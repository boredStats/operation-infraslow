for s = 1:3
    timeseries1 = Localized_Session1.trial{1};
    timeseries2 = Localized_Session2.trial{1};
    timeseries3 = Localized_Session3.trial{1};

    for i = 2:length(Localized_Session1.time)
        epoch = Localized_Session1.trial{i};
        timeseries1 = horzcat(timeseries1, epoch);
    end

    for i = 2:length(Localized_Session2.time)
        epoch = Localized_Session2.trial{i};
        timeseries2 = horzcat(timeseries2, epoch);
    end

    for i = 2:length(Localized_Session3.time)
        epoch = Localized_Session3.trial{i};
        timeseries3 = horzcat(timeseries3, epoch);
    end
end