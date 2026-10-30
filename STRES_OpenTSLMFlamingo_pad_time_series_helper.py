# class OpenTSLMFlamingo(TimeSeriesLLM):
    # ...
    # def pad_and_apply_batch(
    #     self, batch: List[Dict[str, any]], include_labels: bool
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
        def pad_time_series(batch, max_length=None, max_height=None):
            """ 
            Pad time series to the same length and 'height' (either max in batch or specified max), 
            where 'height' refers to the number of channels 
            *** NOTICE (12/11/2025): 
            *** this version of pad_time_series has been adjusted from the original OpenTSLMFlamingo 
            *** implementation so that it also zero-pads in the channel-dimension of multidimensional 
            *** time series 
            """
            time_series = [item["time_series"] for item in batch]

            # Determine target length (either specified or max in batch)
            if max_length is None:
                max_length = max(ts.shape[1] for ts in time_series)
            # Determine target height (either specified or max in batch)
            if max_height is None:
                max_height = max(ts.shape[0] for ts in time_series)

            padded_series = []
            for ts in time_series:
                current_length = ts.shape[1]
                if current_length < max_length:
                    # Pad with zeros to reach max_length
                    # Ensure padding has the same number of dimensions as the time series
                    padding_shape = list(ts.shape)
                    padding_shape[1] = max_length - current_length
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([ts, padding], dim=1)
                else:
                    # If already at or exceeding max_length, truncate
                    padded = ts[:, :max_length]

                current_height = padded.shape[0]
                if current_height < max_height:
                    # Pad with scaled random gaussian noise to reach max_height
                    # Ensure padding has the same number of dimensions as the time series
                    padding_shape = list(padded.shape)
                    padding_shape[0] = max_height - current_height
                    padding = torch.zeros(
                        padding_shape, device=ts.device, dtype=ts.dtype
                    )
                    padded = torch.cat([padded, padding], dim=0)
                else:
                    # If already at or exceeding max_length, truncate
                    padded = padded[:max_height, :]

                padded_series.append(padded)

            return torch.stack(padded_series)
        # ...
    # ...