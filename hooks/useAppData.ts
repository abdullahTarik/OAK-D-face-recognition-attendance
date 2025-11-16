
import { useState, useEffect } from 'react';
import { AppData } from '../types';

export const useAppData = () => {
    const [data, setData] = useState<AppData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('/api/status');
                if (!response.ok) {
                    throw new Error(`Network response was not ok: ${response.status} ${response.statusText}`);
                }
                const jsonData: AppData = await response.json();
                setData(jsonData);
                setLoading(false);
            } catch (err) {
                if (err instanceof Error) {
                    setError(err.message);
                } else {
                    setError('An unknown error occurred');
                }
                setLoading(false);
            }
        };

        fetchData();
        
        // Poll for updates every 5 seconds
        const interval = setInterval(fetchData, 5000);
        
        return () => clearInterval(interval);
    }, []);

    return { data, loading, error };
};
